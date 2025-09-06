import os
import abc
import torch
import accelerate
from torch import nn
import torch.nn.functional as F
from typing import Any, Callable, Optional, Union
from transformers import PreTrainedTokenizer
from .grpo import GRPOTrainer
from .grpo_config import GRPOConfig
from .teacher_base import TeacherReward, TeacherTrainer
from .utils_trl_15 import prepare_deepspeed
from transformers import AutoModelForCausalLM


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
