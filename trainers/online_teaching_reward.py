from .extraction_utils import ATLASExtractionUtils


class OnlineTeachingReward:
    
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
        self.__name__ = 'OnlineTeachingReward'
    
    def evaluate_student_performance(self, question, solution, ground_truth):
        return 1.0 if ATLASExtractionUtils.check_correctness(solution, ground_truth) else 0.0
    
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
            
            baseline_length = len(self.tokenizer.encode(baseline_solution))
            student_length = len(self.tokenizer.encode(solution))
            
            if baseline_length > 0 and student_length < baseline_length:
                efficiency = (baseline_length - student_length) / baseline_length
            else:
                efficiency = 0.0
            
            if performance_with_teaching and not performance_without_teaching:
                reward = 1.0 + efficiency
            elif performance_with_teaching and performance_without_teaching:
                reward = efficiency
            elif not performance_with_teaching and performance_without_teaching:
                reward = -1.0
            else:
                reward = 0.0
            
            rewards.append(reward)
        
        return rewards