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


class AdaptiveTeachingReward(TeacherReward):
    
    def __init__(
        self,
        student_model=None,
        teacher_model=None,
        tokenizer=None,
        degradation_penalty_multiplier=2.0,
        efficiency_weight=1.0,
    ):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.degradation_penalty_multiplier = degradation_penalty_multiplier
        self.efficiency_weight = efficiency_weight
        self.__name__ = 'AdaptiveTeachingReward'
    
    def link_with_trainer(
        self, trainer, student_model, teacher_model, tokenizer,
    ):
        TeacherReward.link_with_trainer(
            self=self,
            trainer=trainer,
            student_model=student_model,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
        )
    
    @torch.no_grad()
    def evaluate_student_performance(self, question, solution, ground_truth):
        if not ground_truth:
            return 0.0
        
        solution_clean = solution.strip()
        ground_truth_clean = ground_truth.strip()
        
        if solution_clean == ground_truth_clean:
            return 1.0
        
        solution_lower = solution_clean.lower()
        ground_truth_lower = ground_truth_clean.lower()
        if solution_lower == ground_truth_lower:
            return 1.0
        
        return 0.0
    
    def __call__(
        self,
        prompts,
        completions,
        **kwargs,
    ):
        rewards = []
        ground_truths = kwargs.get('ground_truths', [])
        baseline_solutions = kwargs.get('baseline_solutions', [])
        solutions = kwargs.get('solutions', [])
        questions = kwargs.get('questions', prompts)
        
        for i, (question, teacher_completion, solution) in enumerate(
            zip(questions, completions, solutions)
        ):
            ground_truth = ground_truths[i] if i < len(ground_truths) else ""
            baseline_solution = baseline_solutions[i] if i < len(baseline_solutions) else ""
            
            if not ground_truth:
                rewards.append(0.0)
                continue
            
            performance_with_teaching = self.evaluate_student_performance(
                question, solution, ground_truth
            )
            
            performance_without_teaching = self.evaluate_student_performance(
                question, baseline_solution, ground_truth
            )
            
            teaching_content = teacher_completion
            teaching_length = len(self.tokenizer.encode(teaching_content))
            
            performance_delta = performance_with_teaching - performance_without_teaching
            
            if performance_delta < 0:
                reward = -self.degradation_penalty_multiplier * abs(performance_delta)
            else:
                efficiency_bonus = 100.0 / (100.0 + teaching_length)
                reward = performance_delta * (1.0 + self.efficiency_weight * efficiency_bonus)
            
            rewards.append(reward)
        
        return rewards