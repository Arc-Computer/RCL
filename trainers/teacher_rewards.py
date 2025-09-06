import os
import abc
import gc
from collections import defaultdict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Callable, Optional, Sequence
from .teacher_base import (
    find_sublist_start_end, extract_and_left_align_from_mask, TeacherReward,
    find_valid_subsequence, find_first_last_one_idxs, log_tensor_info,
    is_tensor, TeacherTrainer,
)
import re
import random


def combine_items(items):

    if isinstance(items[0], torch.Tensor):
        return torch.cat(items, dim=0)

    elif isinstance(items[0], float):
        return items

    elif isinstance(items[0], list):
        return items

    elif isinstance(items[0], dict):
        combined = {}
        for key in items[0]:

            values = [item[key] for item in items]
            combined[key] = combine_items(values)
        return combined
    else:
        return items


def combine_list_elements(list_of_lists):

    n = len(list_of_lists[0])
    result = []
    for i in range(n):
        items = [lst[i] for lst in list_of_lists]
        result.append(combine_items(items))
    return result


def to_torch_tensor(data, device='cpu', dtype=None):

    if isinstance(data, torch.Tensor):
        return data.to(device, dtype=dtype) if dtype else data.to(device)

    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
        return tensor.to(device, dtype=dtype) if dtype else tensor.to(device)

    if isinstance(data, (list, tuple)):
        tensor = torch.tensor(
            data, dtype=dtype) if dtype else torch.tensor(data)
        return tensor.to(device)

    raise TypeError


class TeacherDummyLengthReward(TeacherReward):

    def __init__(
        self,
        student_model=None,
        teacher_model=None,
        tokenizer=None,
        negative=False,
    ):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.negative = negative
        self.__name__ = 'TeacherDummyLengthReward'

    def link_with_trainer(
            self, trainer, student_model, teacher_model, tokenizer,):
        TeacherReward.link_with_trainer(
            self=self,
            trainer=trainer,
            student_model=student_model,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
        )

    def __call__(
        self,
        prompts,
        completions,
        student_system_prompts,
        start_think_teacher_tags,
        end_think_teacher_tags,
        start_think_student_tags,
        end_think_student_tags,
        start_solution_tags,
        end_solution_tags,
        think_prefixes,
        think_solution_delimiters,
        questions,
        solutions,
        **kwargs,
    ):
        rewards = []
        for completion in completions:
            encoding = self.tokenizer(completion)
            reward = len(encoding)
            if self.negative:
                reward = -1*reward
            rewards.append(reward)
        return rewards


class TeacherKLBasedReward(TeacherReward):

    def __init__(
        self,


        student_model: Any = None,
        teacher_model: Any = None,
        tokenizer: Any = None,

        answer_log_prob_coeff: float | list = 1.0,
        kl_penalty_reward_coeff: float | list = 1.0,
        normalize_log_prob_fn: Optional[Callable | str] = 'exp',
        clip_log_prob: Optional[float] = None,
        normalize_kl_fn: Optional[Callable | str] = 'exp_norm',
        clip_kl: Optional[float] = None,

        reduction_log_prob_fn: Callable | str | list = 'mean',
        reduction_kl_fn: Callable | str | list = 'mean',


        use_schulman_kl_estimation: bool = False,

        positive_kl_estimation: bool = False,
        not_matched_penalty: float = -1.0,


        unbias_teacher_log_probs: Optional[bool] = None,


        unbias_student_log_probs_temp: Optional[float] = None,



        include_teacher_think_entropy: Optional[bool] = None,



        correct_generation_coeff: float = 0.0,
        correct_generation_rollouts: int = 8,
        generation_kwargs: dict = {},
        generation_check_stategy: str = 'ground_truth',
        formatting_sub_rewards: list = [],


        evaluate_refined_solution: bool = False,
    ):

        self.student_model = student_model
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        if isinstance(student_model, str):
            raise NotImplementedError

        self.initialize_reward_processing_and_logging(
            answer_log_prob_coeff=answer_log_prob_coeff,
            kl_penalty_reward_coeff=kl_penalty_reward_coeff,
            normalize_log_prob_fn=normalize_log_prob_fn,
            normalize_kl_fn=normalize_kl_fn,
            reduction_log_prob_fn=reduction_log_prob_fn,
            reduction_kl_fn=reduction_kl_fn,
            clip_log_prob=clip_log_prob,
            clip_kl=clip_kl,
        )

        self.use_schulman_kl_estimation = use_schulman_kl_estimation
        self.not_matched_penalty = not_matched_penalty

        self.unbias_teacher_log_probs = unbias_teacher_log_probs
        self.unbias_student_log_probs_temp = unbias_student_log_probs_temp
        if self.unbias_student_log_probs_temp is None:
            self.unbias_student_log_probs_temp = 1
        else:
            assert self.unbias_student_log_probs_temp > 0
        self.include_teacher_think_entropy = include_teacher_think_entropy
        if self.include_teacher_think_entropy is None:

            self.include_teacher_think_entropy = True

        self.correct_generation_coeff = correct_generation_coeff
        self.correct_generation_rollouts = correct_generation_rollouts
        self.generation_kwargs = generation_kwargs
        self.generation_check_stategy = generation_check_stategy
        if self.correct_generation_coeff != 0.0:
            raise NotImplementedError

        self.formatting_sub_rewards = formatting_sub_rewards
        self.evaluate_refined_solution = evaluate_refined_solution
        if self.formatting_sub_rewards or self.evaluate_refined_solution:
            raise NotImplementedError

        self.__name__ = 'TeacherKLBasedReward'

    def used_device(self, ):
        teacher_device = self.teacher_model.device
        return teacher_device
        if str(teacher_device) == 'meta':
            return 'cuda'
        else:
            return teacher_device

    def initialize_reward_processing_and_logging(
            self,
            answer_log_prob_coeff,
            kl_penalty_reward_coeff,
            normalize_log_prob_fn,
            normalize_kl_fn,
            reduction_log_prob_fn,
            reduction_kl_fn,
            clip_log_prob,
            clip_kl,
    ):

        self.answer_log_prob_coeff = answer_log_prob_coeff
        if isinstance(self.answer_log_prob_coeff, Sequence):
            self.use_answer_log_prob_coeff = True
            self.answer_log_prob_coeff = torch.tensor(
                self.answer_log_prob_coeff)
            if self.teacher_model is not None:
                self.answer_log_prob_coeff = self.answer_log_prob_coeff.to(
                    self.teacher_model.device)
            elif self.student_model is not None:
                self.answer_log_prob_coeff = self.answer_log_prob_coeff.to(
                    self.student_model.device)
            assert (self.answer_log_prob_coeff.shape[-1] ==
                    len(reduction_log_prob_fn))
        else:
            self.use_answer_log_prob_coeff = answer_log_prob_coeff > 0
        self.kl_penalty_reward_coeff = kl_penalty_reward_coeff
        if isinstance(self.kl_penalty_reward_coeff, Sequence):
            self.use_kl_penalty_reward_coeff = True
            self.kl_penalty_reward_coeff = torch.tensor(
                self.kl_penalty_reward_coeff)
            if self.teacher_model is not None:
                self.kl_penalty_reward_coeff = self.kl_penalty_reward_coeff.to(
                    self.teacher_model.device)
            elif self.student_model is not None:
                self.kl_penalty_reward_coeff = self.kl_penalty_reward_coeff.to(
                    self.student_model.device)
            assert (self.kl_penalty_reward_coeff.shape[-1] ==
                    len(reduction_kl_fn))
        else:
            self.use_kl_penalty_reward_coeff = kl_penalty_reward_coeff > 0

        self.initialize_reward_processing_fns(
            normalize_log_prob_fn=normalize_log_prob_fn,
            normalize_kl_fn=normalize_kl_fn,
            reduction_log_prob_fn=reduction_log_prob_fn,
            reduction_kl_fn=reduction_kl_fn,
            clip_log_prob=clip_log_prob,
            clip_kl=clip_kl,
        )

    def initialize_reward_processing_fns(
            self,
            normalize_log_prob_fn,
            normalize_kl_fn,
            reduction_log_prob_fn,
            reduction_kl_fn,
            clip_log_prob,
            clip_kl,
    ):

        if clip_log_prob is not None:

            clip_log_prob = -1*clip_log_prob

        self.normalize_log_prob_fn = self._make_normalize_fn(
            normalize_log_prob_fn,
            temp=1,
            clip_min=clip_log_prob,
        )

        self.normalize_kl_fn = self._make_normalize_fn(
            normalize_kl_fn,
            temp=1,
            clip_max=clip_kl,
        )

        self.reduction_log_prob_fn, self.log_lp_names = self._make_reduction_fn(
            reduction_log_prob_fn, function_log_name='answer_log_prob')

        self.reduction_kl_fn, self.log_kl_names = self._make_reduction_fn(
            reduction_kl_fn, function_log_name='reasoning_kl')

        self.initialize_reductions_to_log()

    def initialize_reductions_to_log(self,):
        reductions_to_log = ['mean', 'sum', 'min', 'max', 'median',
                             'first_quartile', 'last_quartile']
        self.log_reduction_kl_fn, self.log_reduction_kl_log_names = (
            self._make_reduction_fn(reductions_to_log,
                                    'unprocessed_thought_kl/'))
        self.log_reduction_prob_fn, self.log_reduction_prob_log_names = (
            self._make_reduction_fn(reductions_to_log,
                                    'unprocessed_answer_log_prob/'))

    def get_components_dictionaries_to_log(
            self, kl, log_probs, teacher_mask, student_solution_masks):
        full_dict = {}
        if kl is not None:
            reduced_kl = self.log_reduction_kl_fn(x=kl, mask=teacher_mask)
            kl_scores_to_log = {n: reduced_kl[..., i] for i, n in
                                enumerate(self.log_reduction_kl_log_names)}
            full_dict.update(kl_scores_to_log)
        if log_probs is not None:
            reduced_log_probs = self.log_reduction_prob_fn(
                x=log_probs, mask=student_solution_masks)
            log_prob_scores_to_log = {n: reduced_log_probs[..., i] for i, n in
                                      enumerate(self.log_reduction_prob_log_names)}
            full_dict.update(log_prob_scores_to_log)
        return full_dict

    def _print_debugging_logs(self, to_print: str):
        self.trainer._print_debugging_logs(to_print=to_print)

    def get_student_chats_and_relevant_num_tokens(
            self,
            completions,
            student_system_prompts,
            questions,
            solutions,
            start_think_teacher_tags,
            end_think_teacher_tags,
            start_think_student_tags,
            end_think_student_tags,
            start_solution_tags,
            end_solution_tags,
            think_prefixes,
            think_solution_delimiters,
    ):

        match_reward = []
        chats = []
        teacher_completion_list = []
        start_end_teacher_thought_idxs_list = []
        start_end_student_thought_idxs_list = []
        start_end_student_solution_idxs_list = []
        chat_iterator = zip(
            completions,
            student_system_prompts,
            questions,
            solutions,
            start_think_teacher_tags,
            end_think_teacher_tags,
            start_think_student_tags,
            end_think_student_tags,
            start_solution_tags,
            end_solution_tags,
            think_prefixes,
            think_solution_delimiters,
        )

        for batch in chat_iterator:

            completion, student_system_prompt, question, solution = batch[:4]

            (start_think_teacher_tag, end_think_teacher_tag,
             start_think_student_tag, end_think_student_tag,
             start_solution_tag, end_solution_tag,) = [
                 re.escape(tag) for tag in batch[4:-2]]

            (start_think_teacher_tag_no_esc, end_think_teacher_no_esc,
             start_think_student_tag_no_esc, end_think_student_tag_no_esc,
             ) = batch[4:8]

            think_prefix, think_solution_delimiter = batch[-2:]
            reward_match = 0.0
            think_pattern = (
                start_think_teacher_tag + r"(.*?)" + end_think_teacher_tag
            )

            teacher_thought_match = re.search(
                think_pattern, completion, flags=re.DOTALL)

            if not teacher_thought_match:

                reward_match += self.not_matched_penalty
                completion = completion + end_think_teacher_no_esc
                teacher_thought_match = re.search(
                    think_pattern, completion, flags=re.DOTALL)
                if not teacher_thought_match:

                    completion = start_think_teacher_tag_no_esc + completion
                    teacher_thought_match = re.search(
                        think_pattern, completion, flags=re.DOTALL)
                    assert teacher_thought_match

            match_reward.append(reward_match)

            start_teacher_thought = teacher_thought_match.start(1)
            end_teacher_thought = teacher_thought_match.end(1)

            completion_tokens = self.tokenizer.encode(completion)

            thought_content = completion[
                start_teacher_thought:end_teacher_thought]
            thought_tokens_orig = self.tokenizer.encode(thought_content)

            thought_tokens = find_valid_subsequence(
                sub=thought_tokens_orig, seq=completion_tokens)

            start_end_teacher_thought_idxs = find_sublist_start_end(
                completion_tokens,
                thought_tokens,
                from_end=True,
                reverse_search=False,
            )
            while start_end_teacher_thought_idxs is None:
                print('Tokenization error: Missing thought tokens in teacher '
                      'completion.')
                print(completion_tokens)
                print(thought_tokens)
                for _ in range(3):
                    print('='*20)
                print('completion')
                print(completion)
                for _ in range(3):
                    print('='*20)
                print('thought_content')
                print(thought_content)
                thought_tokens_orig = thought_tokens_orig[:-1]
                thought_tokens = find_valid_subsequence(
                    sub=thought_tokens_orig, seq=completion_tokens)

                start_end_teacher_thought_idxs = find_sublist_start_end(
                    completion_tokens,
                    thought_tokens,
                    from_end=True,
                    reverse_search=False,
                )
                thought_content = self.tokenizer.decode(thought_tokens_orig)
                completion = (
                    start_think_teacher_tag + thought_content +
                    end_think_teacher_tag
                )

            start_end_teacher_thought_idxs_list.append(
                start_end_teacher_thought_idxs)

            teacher_completion_list.append(completion)

            student_completion = (
                think_prefix + start_think_student_tag_no_esc + thought_content
                + end_think_student_tag_no_esc + think_solution_delimiter
                + solution)

            student_chat_messages = [
                {
                    "role": "system",
                    "content": student_system_prompt,
                },
                {
                    "role": "user",
                    "content": question,
                },
                {
                    "role": "assistant",
                    "content": student_completion,
                },
            ]
            student_chat = self.tokenizer.apply_chat_template(
                student_chat_messages,
                tokenize=False,
                continue_final_message=False,
            )

            student_chat_tokens = self.tokenizer.encode(student_chat)

            start_end_student_thought_idxs = find_sublist_start_end(
                student_chat_tokens,
                thought_tokens,
                from_end=True,
                reverse_search=False,
            )

            if start_end_student_thought_idxs is None:
                print('Tokenization error: Missing thought tokens in student '
                      'chat.')
                print(student_chat_tokens)
                print(thought_tokens)
                raise NotImplementedError

            start_end_student_thought_idxs_list.append(
                start_end_student_thought_idxs)

            solution_pattern = (
                start_solution_tag + r"(.*?)" + end_solution_tag)
            student_solution_match = re.search(
                solution_pattern, solution, flags=re.DOTALL)

            assert student_solution_match

            sol_start = student_solution_match.start(1)
            sol_end = student_solution_match.end(1)

            solution_without_tags = solution[sol_start:sol_end]
            solution_tokens = self.tokenizer.encode(solution_without_tags)
            solution_tokens = find_valid_subsequence(
                sub=solution_tokens, seq=student_chat_tokens,)
            start_end_student_solution_idxs = find_sublist_start_end(
                student_chat_tokens,
                solution_tokens,
                from_end=True,


                reverse_search=True,
            )

            if start_end_student_solution_idxs is None:
                print('Tokenization error: Missing solution tokens in student '
                      'chat.')
                print(student_chat_tokens)
                print(solution_tokens)
                raise NotImplementedError

            start_end_student_solution_idxs_list.append(
                start_end_student_solution_idxs)

            chats.append(student_chat)

        return (chats,
                match_reward,
                teacher_completion_list,
                start_end_teacher_thought_idxs_list,
                start_end_student_thought_idxs_list,
                start_end_student_solution_idxs_list)

    def link_with_trainer(
            self, trainer, student_model, teacher_model, tokenizer,):
        TeacherReward.link_with_trainer(
            self=self,
            trainer=trainer,
            student_model=student_model,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
        )
        if self.unbias_teacher_log_probs is None:
            self.unbias_teacher_log_probs = True

        if self.unbias_teacher_log_probs:
            self.teacher_gen_temperature = trainer.gen_temperature
        else:
            self.teacher_gen_temperature = 1

        if is_tensor(self.kl_penalty_reward_coeff):
            self.kl_penalty_reward_coeff = self.kl_penalty_reward_coeff.to(
                self.teacher_model.device)
        if is_tensor(self.answer_log_prob_coeff):
            self.answer_log_prob_coeff = self.answer_log_prob_coeff.to(
                self.teacher_model.device)

    def get_mask_for_spans(self, start_end_idxs_list, seq_len, device):

        start_end_idxs_list = [
            (s if s >= 0 else seq_len + s, e if e >= 0 else seq_len + e)
            for s, e in start_end_idxs_list
        ]

        bsz = len(start_end_idxs_list)
        positions = torch.arange(seq_len, device=device)
        positions = positions.unsqueeze(dim=0).expand(bsz, seq_len)
        starts = torch.tensor(
            [s for s, _ in start_end_idxs_list], device=device)
        ends = torch.tensor([e for _, e in start_end_idxs_list], device=device)
        mask = ((positions >= starts.unsqueeze(1)) &
                (positions < ends.unsqueeze(1)))
        return mask

    def estimate_kl(self, p_log_probs, q_log_probs, use_schulman_kl_estimation):
        if use_schulman_kl_estimation is None:
            use_schulman_kl_estimation = self.use_schulman_kl_estimation

        kl = p_log_probs - q_log_probs
        if use_schulman_kl_estimation:

            kl = kl - 1 + torch.exp(-kl)
        return kl

    @torch.no_grad()
    def compute_batch_log_probs(
            self, text, student_model=True, cached_log_probs=None, temperature=1.0):

        if student_model:
            model = self.student_model
        else:
            model = self.teacher_model

        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(model.device)

        if cached_log_probs is not None:
            encoding_shape = encoding.input_ids.shape
            expected_num_tokens = encoding_shape[-1] - 1
            assert expected_num_tokens == cached_log_probs.shape[-1]
            cached_log_probs_tensor = to_torch_tensor(
                cached_log_probs, device=model.device)
            return cached_log_probs_tensor.view(
                *encoding_shape[:-1], encoding_shape[-1] - 1)

        outputs = model(**encoding)
        logits = outputs.logits[:, :-1, :]
        labels = encoding.input_ids[:, 1:]
        single_token_log_probs = []

        for i in range(logits.size(0)):
            b_log_probs = F.log_softmax(logits[i]/temperature, dim=-1)
            b_labels = labels[i].unsqueeze(-1)
            b_token_log_probs = b_log_probs.gather(1, b_labels).squeeze(-1)
            single_token_log_probs.append(b_token_log_probs)

        token_log_probs = torch.stack(single_token_log_probs, dim=0)
        return token_log_probs

    @torch.no_grad()
    def compute_batch_log_probs_with_logits(
            self, text, student_model=True, cached_logits=None, temperature=1.0):

        if student_model:
            model = self.student_model
        else:
            model = self.teacher_model

        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(model.device)

        if cached_logits is not None:
            encoding_shape = encoding.input_ids.shape
            expected_num_tokens = encoding_shape[-1] - 1
            assert expected_num_tokens == cached_logits.shape[-1]
            logits = to_torch_tensor(
                cached_logits, device=model.device)
        else:

            outputs = model(**encoding)
            logits = outputs.logits[:, :-1, :]

        labels = encoding.input_ids[:, 1:]
        single_token_log_probs = []

        for i in range(logits.size(0)):
            scaled_logits = logits[i] / temperature
            logits = logits.detach().cpu()
            b_log_probs = F.log_softmax(scaled_logits, dim=-1)
            b_labels = labels[i].unsqueeze(-1)
            b_token_log_probs = b_log_probs.gather(1, b_labels).squeeze(-1)
            single_token_log_probs.append(b_token_log_probs)

        token_log_probs = torch.stack(single_token_log_probs, dim=0)
        return token_log_probs, logits

    @torch.no_grad()
    def compute_split_batch_log_probs(
        self, text, student_model=True, cached_log_probs=None,
        max_sequence_tokens_to_process=4096,
    ):
        if student_model:
            model = self.student_model
        else:
            model = self.teacher_model

        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(model.device)

        if cached_log_probs is not None:
            expected_num_tokens = encoding.shape[-1] - 1
            assert expected_num_tokens == cached_log_probs.shape[-1]
            return to_torch_tensor(cached_log_probs, device=model.device)

        input_ids = encoding.input_ids
        attention_mask = encoding.get('attention_mask', None)
        batch_size, seq_length = input_ids.shape

        token_log_probs_list = []
        offset = 0
        current_pos = 0
        total_valid = seq_length - 1

        multiple_chunks = max_sequence_tokens_to_process < seq_length

        use_cache = False
        past_key_values = None

        while current_pos < seq_length:
            end_pos = min(
                seq_length, current_pos + max_sequence_tokens_to_process)
            if use_cache:
                chunk_ids = input_ids[:, current_pos:end_pos]
            else:
                chunk_ids = input_ids[:, :end_pos]
            if attention_mask is not None:
                chunk_mask = attention_mask[:, :end_pos]
            else:
                chunk_mask = None

            chunk_len = end_pos - current_pos
            outputs = model(
                input_ids=chunk_ids,
                attention_mask=chunk_mask,
                past_key_values=None if current_pos == 0 else past_key_values,
                use_cache=use_cache,
                num_logits_to_keep=chunk_len,
            )
            if hasattr(outputs, 'past_key_values'):
                past_key_values = outputs.past_key_values

            available = total_valid - offset
            used = min(chunk_len, available)

            valid_chunk_logits = outputs.logits[:, :used, :]

            valid_labels = input_ids[:, current_pos+1:current_pos+used+1]

            chunk_log_probs = F.log_softmax(valid_chunk_logits, dim=-1)
            gathered = chunk_log_probs.gather(2, valid_labels.unsqueeze(-1))
            gathered = gathered.squeeze(-1)
            token_log_probs_list.append(gathered)

            outputs = None
            valid_chunk_logits = None
            chunk_log_probs = None
            gc.collect()
            torch.cuda.empty_cache()

            offset += used
            current_pos = end_pos

        if past_key_values is not None:
            del past_key_values
        gc.collect()
        torch.cuda.empty_cache()

        token_log_probs = torch.cat(token_log_probs_list, dim=1)
        return token_log_probs

    def process_single_reward(
            self,
            chat,
            match_reward,
            teacher_completion,
            start_end_teacher_thought_idxs,
            start_end_student_thought_idxs,
            start_end_student_solution_idxs,
            use_schulman_unbiased_estimate=None,
            include_teacher_think_entropy=True,

            return_info_dict=False,
            return_raw_tensors=False,
            cached_student_log_probs=None,
            cached_teacher_log_probs=None,
            cached_thought_tokens_kl=None,):

        self._print_debugging_logs('computing student logprobs')

        student_log_probs = self.compute_batch_log_probs(
            text=chat,
            student_model=True,
            cached_log_probs=cached_student_log_probs,
            temperature=self.unbias_student_log_probs_temp,
        )

        student_device = self.student_model.device

        self._print_debugging_logs('computing student solution masks')
        student_solution_masks = self.get_mask_for_spans(
            start_end_student_solution_idxs,
            seq_len=student_log_probs.shape[-1],
            device=student_device,
        )

        tensors_to_return_dict = dict(chats=chat)
        if return_raw_tensors:
            tensors_to_return_dict.update(dict(
                student_log_probs=student_log_probs.clone().detach().squeeze(
                    dim=0).cpu(),
                student_solution_masks=(
                    student_solution_masks.clone().detach().squeeze(
                        dim=0).cpu()),
            ))

        if self.use_answer_log_prob_coeff:
            self._print_debugging_logs('processing student logprobs')
            processed_log_probs = self.normalize_log_prob_fn(
                x=student_log_probs)

            log_prob_scores = self.reduction_log_prob_fn(
                x=processed_log_probs, mask=student_solution_masks)

            log_prob_scores = torch.nan_to_num(
                log_prob_scores,
                nan=self.not_matched_penalty,
            )

            log_prob_reward = (
                log_prob_scores*self.answer_log_prob_coeff).sum(-1)

        else:
            raise NotImplementedError

        unprocessed_dict = self.get_components_dictionaries_to_log(
            kl=None,
            log_probs=student_log_probs,
            teacher_mask=None,
            student_solution_masks=student_solution_masks,
        )

        self._print_debugging_logs('computing student thought masks')
        student_thought_masks = self.get_mask_for_spans(
            start_end_student_thought_idxs,
            seq_len=student_log_probs.shape[-1],
            device=student_device,
        )
        student_log_probs, student_mask = extract_and_left_align_from_mask(
            student_log_probs, student_thought_masks)

        if self.use_kl_penalty_reward_coeff or return_raw_tensors:

            teacher_device = self.teacher_model.device

            self._print_debugging_logs('computing teacher log probs')

            reused_cached_kl = cached_thought_tokens_kl is None

            reused_cached_kl = False
            if not reused_cached_kl:
                teacher_log_probs = self.compute_batch_log_probs(
                    text=teacher_completion, student_model=False,
                    cached_log_probs=cached_teacher_log_probs,
                    temperature=self.teacher_gen_temperature,
                )

                if return_raw_tensors:
                    tensors_to_return_dict.update(dict(
                        teacher_log_probs=(
                            teacher_log_probs.clone().detach().squeeze(
                                dim=0).cpu()),
                    ))

                self._print_debugging_logs(
                    'computing teacher thought masks')
                teacher_thought_masks = self.get_mask_for_spans(
                    start_end_teacher_thought_idxs,
                    seq_len=teacher_log_probs.shape[-1],
                    device=teacher_device,
                )

                self._print_debugging_logs('aligning and extracting tokens')

                teacher_log_probs, teacher_mask = (
                    extract_and_left_align_from_mask(
                        teacher_log_probs, teacher_thought_masks))

                self._print_debugging_logs('computing KL')
                thought_tokens_kl = self.estimate_kl(
                    p_log_probs=teacher_log_probs,
                    q_log_probs=student_log_probs,
                    use_schulman_kl_estimation=(
                        use_schulman_unbiased_estimate),
                )

                assert torch.all(teacher_mask == student_mask)

            else:
                thought_tokens_kl = to_torch_tensor(
                    cached_thought_tokens_kl, device=student_log_probs.device)
                assert (
                    thought_tokens_kl.shape[-1] == student_log_probs.shape[-1])
                thought_tokens_kl = thought_tokens_kl.view_as(
                    student_log_probs)
                teacher_mask = student_mask

            unprocessed_dict.update(self.get_components_dictionaries_to_log(


                kl=thought_tokens_kl,
                log_probs=None,
                teacher_mask=teacher_mask,
                student_solution_masks=None,
            ))

            if return_raw_tensors:
                tensors_to_return_dict.update(dict(
                    thought_tokens_kl=(
                        thought_tokens_kl.clone().detach().squeeze(
                            dim=0).cpu()),
                    teacher_mask=teacher_mask.clone().detach().squeeze(
                        dim=0).cpu(),
                ))

            self._print_debugging_logs('processing KL')
            processed_kl = self.normalize_kl_fn(x=thought_tokens_kl)
            kl_scores = self.reduction_kl_fn(x=processed_kl, mask=teacher_mask)
            kl_scores = torch.nan_to_num(
                kl_scores,
                nan=-1*self.not_matched_penalty,
            )
            kl_reward = (kl_scores*self.kl_penalty_reward_coeff).sum(-1)
        else:
            thought_tokens_kl = None
            processed_kl = 0.0
            kl_scores = 0.0
            kl_reward = torch.zeros_like(log_prob_reward)

        kl_reward = kl_reward*-1
        match_reward = torch.tensor(
            match_reward, device=log_prob_reward.device)

        assert log_prob_reward.shape == match_reward.shape
        assert kl_reward.shape == match_reward.shape
        reward = log_prob_reward + kl_reward + match_reward

        log_prob_scores_to_log = {
            n: log_prob_scores[..., i] for i, n in enumerate(self.log_lp_names)}
        kl_scores_to_log = {
            n: kl_scores[..., i] for i, n in enumerate(self.log_kl_names)}

        processed_kl_no_entropy = self.normalize_kl_fn(x=-student_log_probs)
        kl_scores_no_entropy = self.reduction_kl_fn(
            x=processed_kl_no_entropy, mask=student_mask)
        kl_scores_no_entropy = torch.nan_to_num(
            kl_scores_no_entropy,
            nan=self.not_matched_penalty,
        )
        kl_reward_no_entropy = -1*(
            kl_scores_no_entropy*self.kl_penalty_reward_coeff).sum(-1)
        reward_no_entropy = (
            log_prob_reward + kl_reward_no_entropy + match_reward)

        kl_dict_no_entropy = self.get_components_dictionaries_to_log(
            kl=-student_log_probs,
            log_probs=None,
            teacher_mask=teacher_mask,
            student_solution_masks=None,
        )
        kl_dict_no_entropy = {
            f'no_entropy_{k}': v for k, v in kl_dict_no_entropy.items()}

        kl_scores_no_entropy_to_log = {
            f'no_entropy_{n}': kl_scores_no_entropy[..., i]
            for i, n in enumerate(self.log_kl_names)}

        self._print_debugging_logs('logging')
        logged_dict = self.trainer.log_metric(
            **unprocessed_dict,
            **log_prob_scores_to_log,
            solution_log_prob_reward=log_prob_reward,
            **kl_scores_to_log,
            thought_processed_kl=processed_kl,
            thought_kl_scores=kl_scores,
            kl_reward=kl_reward,
            match_reward=match_reward,
            total_teacher_likelihood_reward=reward,

            processed_kl_no_entropy=processed_kl_no_entropy,
            kl_scores_no_entropy=kl_scores_no_entropy,
            kl_reward_no_entropy=kl_reward_no_entropy,
            total_tl_reward_no_entropy=reward_no_entropy,
            **kl_dict_no_entropy,
            **kl_scores_no_entropy_to_log,
        )
        if self.trainer.accelerator.is_main_process:
            self._print_debugging_logs('logging chats')
            for chat in chat:
                self.trainer.log_to_file(chat)

        if include_teacher_think_entropy:
            rw_list = reward.tolist()
        else:
            rw_list = reward_no_entropy.tolist()
        if return_info_dict or return_raw_tensors:
            return rw_list, logged_dict, tensors_to_return_dict
        return rw_list

    @torch.no_grad()
    def __call__(
            self,
            prompts,
            completions,
            student_system_prompts,
            start_think_teacher_tags,
            end_think_teacher_tags,
            start_think_student_tags,
            end_think_student_tags,
            start_solution_tags,
            end_solution_tags,
            think_prefixes,
            think_solution_delimiters,
            questions,
            solutions,
            masked_out_think_prefix,
            use_schulman_unbiased_estimate=None,
            include_teacher_think_entropy=None,

            return_info_dict=False,
            return_raw_tensors=False,
            cached_student_log_probs=None,
            cached_teacher_log_probs=None,
            cached_thought_tokens_kl=None,
            **kwargs,):

        if include_teacher_think_entropy is None:
            include_teacher_think_entropy = self.include_teacher_think_entropy

        full_teacher_solutions = [
            p + c for p, c in zip(prompts, completions)]

        completions = [
            p + c for p, c in zip(masked_out_think_prefix, completions)]

        prompts_no_prefix = [
            p.removesuffix(pre) for pre, p in zip(
                masked_out_think_prefix, prompts)
        ]

        self._print_debugging_logs('inside teacher reward, extracting chats')
        (chats,
         match_reward,
         teacher_completion_list,
         start_end_teacher_thought_idxs_list,
         start_end_student_thought_idxs_list,
         start_end_student_solution_idxs_list) = (
            self.get_student_chats_and_relevant_num_tokens(
                completions=completions,
                student_system_prompts=student_system_prompts,
                questions=questions,
                solutions=solutions,
                start_think_teacher_tags=start_think_teacher_tags,
                end_think_teacher_tags=end_think_teacher_tags,
                start_think_student_tags=start_think_student_tags,
                end_think_student_tags=end_think_student_tags,
                start_solution_tags=start_solution_tags,
                end_solution_tags=end_solution_tags,
                think_prefixes=think_prefixes,
                think_solution_delimiters=think_solution_delimiters,
            )
        )

        rec_teacher_solutions = [
            p + c for p, c in zip(
                prompts_no_prefix, teacher_completion_list)]

        for i, (ft, rec_ft, mr) in enumerate(zip(
                full_teacher_solutions, rec_teacher_solutions, match_reward)):

            teacher_se = start_end_teacher_thought_idxs_list[i]
            teacher_enc = self.tokenizer(
                rec_ft,
                return_tensors='pt',
                padding=True,
                truncation=True
            )['input_ids'][0]
            teacher_enc_f = teacher_enc[teacher_se[0]:teacher_se[1]]
            student_se = start_end_student_thought_idxs_list[i]
            student_enc = self.tokenizer(
                chats[i],
                return_tensors='pt',
                padding=True,
                truncation=True
            )['input_ids'][0]
            student_enc_f = student_enc[student_se[0]:student_se[1]]
            if not torch.all(student_enc_f == teacher_enc_f):
                print(
                    f'Warning - student teacher tokens, match reward sc {mr}')
            teacher_thought_masks = self.get_mask_for_spans(
                [teacher_se],
                seq_len=teacher_enc.shape[-1],
                device=teacher_enc.device,
            )

            teacher_enc_al, teacher_mask = (
                extract_and_left_align_from_mask(
                    teacher_enc.unsqueeze(0), teacher_thought_masks))

            if not torch.all(teacher_enc_al == teacher_enc_f):
                print(f'Warning - aligned and extracted encs, mismatch'
                      f' match reward sc {mr}')

        num_chats = len(chats)
        out_values = []
        for i in range(num_chats):
            out = self.process_single_reward(
                chat=chats[i:i+1],
                match_reward=match_reward[i:i+1],

                teacher_completion=rec_teacher_solutions[i:i+1],
                start_end_teacher_thought_idxs=(
                    start_end_teacher_thought_idxs_list[i:i+1]),
                start_end_student_thought_idxs=(
                    start_end_student_thought_idxs_list[i:i+1]),
                start_end_student_solution_idxs=(
                    start_end_student_solution_idxs_list[i:i+1]),
                use_schulman_unbiased_estimate=(
                    use_schulman_unbiased_estimate),
                include_teacher_think_entropy=include_teacher_think_entropy,
                return_info_dict=return_info_dict,
                return_raw_tensors=return_raw_tensors,

                cached_student_log_probs=(
                    cached_student_log_probs[i]
                    if cached_student_log_probs is not None else None),
                cached_teacher_log_probs=(
                    cached_teacher_log_probs[i]
                    if cached_teacher_log_probs is not None else None),
                cached_thought_tokens_kl=(
                    cached_thought_tokens_kl[i]
                    if cached_thought_tokens_kl is not None else None),
            )
            out_values.append(out)
            gc.collect()
            torch.cuda.empty_cache()

        rw_list = []
        if return_info_dict or return_raw_tensors:
            logged_dicts, tensors_to_return_dicts = [], []
            for out in out_values:
                rw, logged_dict, tensors_to_return_dict = out
                rw_list += rw
                logged_dicts.append(logged_dict)
                tensors_to_return_dicts.append(tensors_to_return_dict)
            return rw_list, logged_dicts, tensors_to_return_dicts
        for rw in out_values:
            rw_list += rw
        return rw_list
