import os
import abc
import gc
import copy
import torch
import accelerate
from torch import nn
import torch.nn.functional as F
from typing import Any, Callable, Optional, Union
from transformers import PreTrainedTokenizer
from .grpo import GRPOTrainer
from .grpo_config import GRPOConfig
from .teacher_base import TeacherReward, TeacherTrainer
from .utils_trl_15 import prepare_deepspeed, unwrap_model_for_generation
from transformers import AutoModelForCausalLM
from trl.data_utils import maybe_apply_chat_template, is_conversational
from accelerate.utils import gather, gather_object


class TeacherGRPOTrainer(GRPOTrainer, TeacherTrainer):
    def __init__(
            self,
            *args,
            student_model=None,


            use_reference_teacher_model=False,
            student_model_init_kwargs=None,
            logging_prob=0.0,


            disable_student_offloading=False,
            **kwargs):

        GRPOTrainer.__init__(self, *args, **kwargs)
        if student_model_init_kwargs is None:
            student_model_init_kwargs = self._stored_model_init_kwargs

        offload_student_model = self.offload_untrained_models and (
            not disable_student_offloading)
        if student_model is None:

            self.student_model = self.ref_model
        elif isinstance(student_model, str):
            self.student_model = AutoModelForCausalLM.from_pretrained(
                student_model, **student_model_init_kwargs)
            if self.is_deepspeed_enabled:
                self.student_model = prepare_deepspeed(
                    self.student_model,
                    self.accelerator,
                    offload_to_cpu=offload_student_model)
            else:
                self.student_model = self.accelerator.prepare_model(
                    self.student_model, evaluation_mode=True)

                if offload_student_model:
                    self.student_model = accelerate.cpu_offload(
                        model=self.student_model)
        else:

            raise NotImplementedError
            self.student_model = student_model

        if use_reference_teacher_model:
            teacher_model = self.ref_model
        else:
            teacher_model = self.model

        TeacherTrainer.__init__(
            self,
            student_model=self.student_model,
            teacher_model=teacher_model,
            tokenizer=self.processing_class,
            reward_functions=self.reward_funcs,
            output_dir=self.args.output_dir,
            logging_prob=logging_prob,
        )
        
        self.max_probe_tokens = getattr(self.args, 'max_probe_tokens', 500)
        self.student_diagnostic_template = getattr(self.args, 'student_diagnostic_template',
                                                   "Question: {question}\n\nBriefly outline your approach:")
        self.teacher_adaptive_template = getattr(self.args, 'teacher_adaptive_template',
                                                "Question: {question}\n\nStudent's approach: {approach}\n\nProvide adaptive teaching:")
        self.student_with_teaching_template = getattr(self.args, 'student_with_teaching_template',
                                                     "Question: {question}\n\nTeaching: {teaching}\n\nSolve the problem:")
        self.student_baseline_template = getattr(self.args, 'student_baseline_template',
                                                "Question: {question}\n\nSolve this problem:")
    
    def _generate_and_score(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)[
            "prompt"] for example in inputs]
        
        self._print_debugging_logs('Generating baseline solutions...')
        baseline_completions_text = self._generate_baseline_solutions(prompts_text)
        
        self._print_debugging_logs('Generating student approaches...')
        student_approaches = self._generate_student_approaches(prompts_text)
        
        self._print_debugging_logs('Generating teacher adaptive teaching...')
        teacher_prompts = []
        for prompt, approach in zip(prompts_text, student_approaches):
            teacher_prompt = self.teacher_adaptive_template.format(
                question=prompt, approach=approach
            )
            teacher_prompts.append(teacher_prompt)
        
        if self.args.use_vllm or self.args.use_ray or self.args.use_vllm_server:
            teacher_completion_ids = self._generate_with_vllm(teacher_prompts)
        else:
            teacher_completion_ids = self._generate_with_model(self.model, teacher_prompts)
        
        teacher_completions_text = self.processing_class.batch_decode(
            teacher_completion_ids, skip_special_tokens=True)
        
        self._print_debugging_logs('Generating student solutions with teaching...')
        student_prompts_with_teaching = []
        for prompt, teaching in zip(prompts_text, teacher_completions_text):
            student_prompt = self.student_with_teaching_template.format(
                question=prompt, teaching=teaching
            )
            student_prompts_with_teaching.append(student_prompt)
        
        if self.args.use_vllm or self.args.use_ray or self.args.use_vllm_server:
            student_completion_ids = self._generate_with_vllm(student_prompts_with_teaching)
        else:
            student_completion_ids = self._generate_with_model(self.student_model, student_prompts_with_teaching)
        
        student_completions_text = self.processing_class.batch_decode(
            student_completion_ids, skip_special_tokens=True)
        
        student_solutions = self._extract_solutions_from_responses(student_completions_text)
        
        completions = teacher_completions_text
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, teacher_completions_text):
                bootstrap = prompt.pop()[
                    "content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}])
        
        self._print_debugging_logs('computing rewards...')
        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device)
        
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c}
                                for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)[
                        "text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(
                        **reward_inputs).logits[:, 0]
            else:
                keys = [key for key in inputs[0]
                        if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key]
                                     for example in inputs] for key in keys}
                
                reward_kwargs['baseline_solutions'] = baseline_completions_text
                reward_kwargs['solutions'] = student_solutions
                
                output_reward_func = reward_func(
                    prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(
                    output_reward_func, dtype=torch.float32, device=device)
        
        self._print_debugging_logs('gathering rewards...')
        
        rewards_per_func = gather(rewards_per_func)
        
        rewards = (rewards_per_func *
                   self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        
        self._print_debugging_logs('normalizing rewards...')
        
        mean_grouped_rewards = rewards.view(-1,
                                           self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / \
            (std_grouped_rewards + 1e-4)
        
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        
        return prompts_text, teacher_completion_ids, advantages
    
    def _extract_solutions_from_responses(self, responses):
        import re
        solutions = []
        for response in responses:
            match = re.search(r'<solution>(.*?)</solution>', response, re.DOTALL)
            if match:
                solutions.append(match.group(1).strip())
            else:
                solutions.append("")
        return solutions
    
    def _generate_baseline_solutions(self, prompts_text):
        self._print_debugging_logs('Generating standalone student solutions...')
        
        baseline_prompts = []
        for prompt in prompts_text:
            baseline_prompt = self.student_baseline_template.format(question=prompt)
            baseline_prompts.append(baseline_prompt)
        
        if self.args.use_vllm or self.args.use_ray or self.args.use_vllm_server:
            completion_ids = self._generate_with_vllm(baseline_prompts)
        else:
            completion_ids = self._generate_with_model(self.student_model, baseline_prompts)
        
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True)
        
        solutions = self._extract_solutions_from_responses(completions_text)
        
        return solutions
    
    def _generate_student_approaches(self, prompts_text):
        probe_prompts = []
        for prompt in prompts_text:
            probe_prompt = self.student_diagnostic_template.format(question=prompt)
            probe_prompts.append(probe_prompt)
        
        if self.args.use_vllm or self.args.use_ray or self.args.use_vllm_server:
            approach_ids = self._generate_with_vllm(probe_prompts, max_tokens=self.max_probe_tokens)
        else:
            approach_ids = self._generate_with_model(self.student_model, probe_prompts, 
                                                    max_tokens=self.max_probe_tokens)
        
        approaches_text = self.processing_class.batch_decode(
            approach_ids, skip_special_tokens=True)
        
        return approaches_text
    
    def _generate_with_model(self, model, prompts_text, max_tokens=None):
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, 
            padding_side="left", add_special_tokens=False
        )
        
        if hasattr(model, 'hf_device_map'):
            device = list(model.hf_device_map.values())[0]
        else:
            device = next(model.parameters()).device
        
        prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
        
        generation_config = self.generation_config
        if max_tokens is not None:
            generation_config = copy.deepcopy(self.generation_config)
            generation_config.max_new_tokens = max_tokens
        
        with torch.inference_mode():
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, 
                    generation_config=generation_config
                )
        
        prompt_length = prompt_ids.size(1)
        completion_ids = torch.unbind(
            prompt_completion_ids[:, prompt_length:].cpu(), dim=0)
        
        return completion_ids
    
    def _generate_with_vllm(self, prompts_text, max_tokens=None):
        if max_tokens is None:
            max_tokens = self.max_completion_length
        
        raise NotImplementedError("vLLM generation for adaptive teaching not yet implemented")
    
