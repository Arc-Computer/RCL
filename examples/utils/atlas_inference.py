"""
ATLAS Two-Pass Inference Protocol Implementation

Core logic for diagnostic probing and adaptive teaching.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Tuple
import json


class ATLASInference:
    """
    ATLAS two-pass inference protocol implementation.
    
    Combines diagnostic probing with adaptive teaching to improve
    student model performance across various tasks.
    """
    
    def __init__(
        self,
        student_model,
        student_tokenizer,
        teacher_model,
        teacher_tokenizer,
        probe_token_limit: int = 50,
        device: str = "auto"
    ):
        self.student_model = student_model
        self.student_tokenizer = student_tokenizer
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer
        self.probe_token_limit = probe_token_limit
        self.device = device
    
    def diagnostic_probe(self, problem: str) -> Dict[str, any]:
        """
        Phase 1: Assess student understanding with minimal interaction.
        
        Args:
            problem: The input problem/question
            
        Returns:
            Dict containing capability assessment and probe response
        """
        # Create diagnostic prompt for teacher
        probe_prompt = f"""Assess the difficulty of this problem for a 4B parameter model and predict their capability level.

Problem: {problem}

Rate the student's likely performance (1-5 scale):
1. Will likely fail completely
2. May attempt but make significant errors  
3. Mixed performance, some correct reasoning
4. Will likely succeed with minor errors
5. Should handle this easily

Provide assessment:"""

        # Generate teacher's diagnostic assessment
        messages = [{"role": "user", "content": probe_prompt}]
        inputs = self.teacher_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.teacher_model.device)
        
        with torch.no_grad():
            outputs = self.teacher_model.generate(
                **inputs,
                max_new_tokens=self.probe_token_limit,
                temperature=0.1,
                do_sample=False
            )
        
        probe_response = self.teacher_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        
        # Extract capability score (simple heuristic)
        capability_score = self._extract_capability_score(probe_response)
        
        return {
            "capability_score": capability_score,
            "probe_response": probe_response,
            "teaching_strategy": self._determine_teaching_strategy(capability_score)
        }
    
    def adaptive_teaching(
        self, 
        problem: str, 
        diagnostic_result: Dict[str, any]
    ) -> Dict[str, any]:
        """
        Phase 2: Provide capability-adapted teaching guidance.
        
        Args:
            problem: The input problem/question
            diagnostic_result: Result from diagnostic_probe
            
        Returns:
            Dict containing teaching guidance and strategy
        """
        capability_score = diagnostic_result["capability_score"]
        strategy = diagnostic_result["teaching_strategy"]
        
        if strategy == "Light":
            teaching_prompt = f"""The student model should handle this well. Provide minimal intervention - just a brief hint or confirmation of approach.

Problem: {problem}

Brief guidance:"""
        
        elif strategy == "Medium":
            teaching_prompt = f"""The student model may need guided discovery. Provide structured hints and outline key steps without giving away the complete solution.

Problem: {problem}

Structured guidance:"""
        
        else:  # Heavy
            teaching_prompt = f"""The student model will likely struggle. Provide direct support with step-by-step breakdown, examples, and clear explanations.

Problem: {problem}

Comprehensive teaching:"""
        
        # Generate teaching guidance
        messages = [{"role": "user", "content": teaching_prompt}]
        inputs = self.teacher_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.teacher_model.device)
        
        with torch.no_grad():
            outputs = self.teacher_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True
            )
        
        teaching_response = self.teacher_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        
        return {
            "teaching_guidance": teaching_response,
            "strategy": strategy,
            "capability_score": capability_score
        }
    
    def generate_student_response(
        self, 
        problem: str, 
        teaching_guidance: Optional[str] = None
    ) -> str:
        """
        Generate student response with or without teacher guidance.
        
        Args:
            problem: The input problem/question
            teaching_guidance: Optional teaching guidance from teacher
            
        Returns:
            Student's generated response
        """
        if teaching_guidance:
            # Guided generation
            prompt = f"""Problem: {problem}

Guidance: {teaching_guidance}

Solution:"""
        else:
            # Baseline generation
            prompt = f"""Problem: {problem}

Solution:"""
        
        messages = [{"role": "user", "content": prompt}]
        inputs = self.student_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.student_model.device)
        
        with torch.no_grad():
            outputs = self.student_model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.student_tokenizer.eos_token_id
            )
        
        response = self.student_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def run_full_protocol(self, problem: str) -> Dict[str, any]:
        """
        Run complete ATLAS two-pass protocol.
        
        Args:
            problem: The input problem/question
            
        Returns:
            Complete results including diagnostic, teaching, and response
        """
        # Phase 1: Diagnostic probing
        diagnostic_result = self.diagnostic_probe(problem)
        
        # Phase 2: Adaptive teaching
        teaching_result = self.adaptive_teaching(problem, diagnostic_result)
        
        # Generate baseline (student alone)
        baseline_response = self.generate_student_response(problem)
        
        # Generate guided response (student + teacher)
        guided_response = self.generate_student_response(
            problem, 
            teaching_result["teaching_guidance"]
        )
        
        return {
            "problem": problem,
            "diagnostic": diagnostic_result,
            "teaching": teaching_result,
            "baseline_response": baseline_response,
            "guided_response": guided_response
        }
    
    def _extract_capability_score(self, probe_response: str) -> int:
        """Extract capability score from probe response with robust parsing.
        
        Looks for various patterns indicating capability level:
        - Direct numbers (1-5)
        - Word mappings (weak, moderate, strong)
        - Percentage confidence levels
        """
        import re
        
        response_lower = probe_response.lower()
        
        # Direct number patterns
        patterns = [
            (r'\b([1-5])\s*(?:out of 5|/5|\s+score)', lambda m: int(m.group(1))),
            (r'score[:\s]+([1-5])\b', lambda m: int(m.group(1))),
            (r'level[:\s]+([1-5])\b', lambda m: int(m.group(1))),
            (r'capability[:\s]+([1-5])\b', lambda m: int(m.group(1))),
        ]
        
        for pattern, extractor in patterns:
            match = re.search(pattern, response_lower)
            if match:
                return extractor(match)
        
        # Word-based mappings
        if any(word in response_lower for word in ["very weak", "will fail", "struggle significantly"]):
            return 1
        elif any(word in response_lower for word in ["weak", "likely fail", "major errors"]):
            return 2
        elif any(word in response_lower for word in ["moderate", "mixed", "some errors"]):
            return 3
        elif any(word in response_lower for word in ["strong", "likely succeed", "minor errors"]):
            return 4
        elif any(word in response_lower for word in ["very strong", "easily", "no issues"]):
            return 5
        
        # Fallback to simple number search
        for i in range(5, 0, -1):
            if str(i) in probe_response:
                return i
        
        return 3  # Default to middle score
    
    def _determine_teaching_strategy(self, capability_score: int) -> str:
        """Determine teaching strategy based on capability score.
        
        Maps capability scores to teaching strategies:
        - Scores 4-5: Light intervention (high capability)
        - Scores 2-3: Medium guidance (moderate capability)  
        - Score 1: Heavy support (low capability)
        """
        if capability_score >= 4:
            return "Light"
        elif capability_score >= 2:
            return "Medium" 
        else:
            return "Heavy"


def load_atlas_models(
    student_model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    teacher_thinking_name: str = "Arc-Intelligence/ATLAS-8B-Thinking",
    teacher_instruct_name: str = "Arc-Intelligence/ATLAS-8B-Instruct",
    device_map: str = "auto",
    torch_dtype = torch.float16
) -> Tuple[ATLASInference, ATLASInference]:
    """
    Load ATLAS models for both reasoning and code generation tasks.
    
    Returns:
        Tuple of (reasoning_atlas, code_atlas) inference objects
    """
    # Check GPU memory before loading
    if torch.cuda.is_available():
        memory_available = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory Available: {memory_available:.1f} GB")
        if memory_available < 12:
            print("⚠️  Warning: Limited GPU memory. Consider using smaller models or quantization.")
            print("   Recommended: Use Google Colab with T4 (16GB) or A100 (40GB) GPU")
    
    try:
        print("Loading student model...")
        student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        student_model = AutoModelForCausalLM.from_pretrained(
            student_model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
    except torch.cuda.OutOfMemoryError:
        print("❌ GPU Out of Memory Error!")
        print("Solutions:")
        print("1. Free GPU memory: torch.cuda.empty_cache()")
        print("2. Use smaller batch size")
        print("3. Enable 8-bit quantization: load_in_8bit=True")
        print("4. Use CPU fallback: device_map='cpu'")
        raise
    except Exception as e:
        if "CUDA out of memory" in str(e):
            print("❌ GPU memory exhausted during model loading")
            print("Try using bitsandbytes for 8-bit quantization:")
            print("pip install bitsandbytes")
            print("Then add: load_in_8bit=True to model loading")
        raise
    
    print("Loading ATLAS-8B-Thinking teacher...")
    thinking_tokenizer = AutoTokenizer.from_pretrained(teacher_thinking_name)
    thinking_model = AutoModelForCausalLM.from_pretrained(
        teacher_thinking_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    
    print("Loading ATLAS-8B-Instruct teacher...")
    instruct_tokenizer = AutoTokenizer.from_pretrained(teacher_instruct_name)
    instruct_model = AutoModelForCausalLM.from_pretrained(
        teacher_instruct_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    
    # Create inference objects
    reasoning_atlas = ATLASInference(
        student_model=student_model,
        student_tokenizer=student_tokenizer,
        teacher_model=thinking_model,
        teacher_tokenizer=thinking_tokenizer
    )
    
    code_atlas = ATLASInference(
        student_model=student_model,
        student_tokenizer=student_tokenizer,
        teacher_model=instruct_model,
        teacher_tokenizer=instruct_tokenizer
    )
    
    print("Models loaded successfully!")
    return reasoning_atlas, code_atlas