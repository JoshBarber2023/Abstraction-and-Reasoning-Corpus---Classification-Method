import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import time
from openai import OpenAI
import re
from pathlib import Path
import traceback

class SolutionValidator:
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.validation_attempts = {}
        self.max_attempts_per_hypothesis = 3
        self.max_validation_loops = 2
    
    def validate_hypothesis_by_solving(self, hypothesis: Dict[str, Any], 
                                     training_pairs: List[Tuple[np.ndarray, np.ndarray]],
                                     task_name: str) -> Dict[str, Any]:
        """
        Validate a hypothesis by actually trying to solve the transformation
        """
        if len(training_pairs) < 2:
            # Need at least 2 examples for validation
            hypothesis['validation_result'] = 'insufficient_data'
            hypothesis['validation_score'] = 0.0
            return hypothesis
            
        validation_key = f"{task_name}_{hypothesis.get('description', '')[:50]}"
        
        if validation_key not in self.validation_attempts:
            self.validation_attempts[validation_key] = {
                'attempts': 0,
                'failed_approaches': [],
                'successful_approach': None
            }
        
        attempts_data = self.validation_attempts[validation_key]
        
        if attempts_data['successful_approach']:
            # Already found working approach
            hypothesis['validation_result'] = 'already_validated'
            hypothesis['validation_score'] = 1.0
            hypothesis['working_approach'] = attempts_data['successful_approach']
            return hypothesis
            
        if attempts_data['attempts'] >= self.max_attempts_per_hypothesis:
            hypothesis['validation_result'] = 'max_attempts_exceeded'
            hypothesis['validation_score'] = 0.0
            return hypothesis
        
        # Try to solve using this hypothesis
        success_count = 0
        total_tests = min(len(training_pairs), 3)  # Test up to 3 pairs
        
        for test_idx in range(total_tests):
            # Use all other pairs as training, test on this one
            train_pairs = [training_pairs[i] for i in range(len(training_pairs)) if i != test_idx]
            test_input, expected_output = training_pairs[test_idx]
            
            # Attempt to solve
            solution_result = self._attempt_solution(
                hypothesis, train_pairs, test_input, expected_output, 
                attempts_data['failed_approaches']
            )
            
            if solution_result['success']:
                success_count += 1
                if not attempts_data['successful_approach']:
                    attempts_data['successful_approach'] = solution_result['approach']
        
        attempts_data['attempts'] += 1
        validation_score = success_count / total_tests
        
        hypothesis['validation_result'] = 'tested'
        hypothesis['validation_score'] = validation_score
        hypothesis['validation_success_rate'] = f"{success_count}/{total_tests}"
        
        if validation_score >= 0.8:
            hypothesis['validation_passed'] = True
        else:
            hypothesis['validation_passed'] = False
            
        return hypothesis
    
    def _attempt_solution(self, hypothesis: Dict[str, Any], 
                         training_pairs: List[Tuple[np.ndarray, np.ndarray]],
                         test_input: np.ndarray, expected_output: np.ndarray,
                         failed_approaches: List[str]) -> Dict[str, Any]:
        """
        Attempt to solve a single test case using the hypothesis
        """
        
        # Prepare training context
        training_context = self._format_training_context(training_pairs)
        failed_context = "\n".join([f"- {approach}" for approach in failed_approaches[-3:]])  # Last 3 failures
        
        prompt = f"""You are solving an ARC transformation puzzle. Use the hypothesis and training examples to transform the test input.

HYPOTHESIS: {hypothesis['description']}
CATEGORY: {hypothesis['category']}
EVIDENCE: {hypothesis.get('evidence', 'None provided')}

TRAINING EXAMPLES:
{training_context}

PREVIOUSLY FAILED APPROACHES:
{failed_context}

TEST INPUT TO TRANSFORM:
{self._grid_to_string(test_input)}

Think step by step:
1. Analyze how the hypothesis applies to the training examples
2. Identify the specific transformation rule
3. Apply this rule to the test input
4. Generate the output grid

Provide your response as:
REASONING: [Your step-by-step analysis]
TRANSFORMATION_RULE: [Precise rule derived from hypothesis]
OUTPUT_GRID: [The transformed grid as a JSON array]
CONFIDENCE: [0.0 to 1.0]"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Solve ARC puzzles by applying hypotheses to transform grids. Be precise and logical."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            content = response.choices[0].message.content
            
            # Parse response
            reasoning = self._extract_section(content, "REASONING:")
            transformation_rule = self._extract_section(content, "TRANSFORMATION_RULE:")
            output_grid_text = self._extract_section(content, "OUTPUT_GRID:")
            confidence = self._extract_section(content, "CONFIDENCE:")
            
            # Try to parse the output grid
            predicted_output = self._parse_grid_from_text(output_grid_text)
            
            if predicted_output is not None:
                # Check if solution matches expected output
                if np.array_equal(predicted_output, expected_output):
                    return {
                        'success': True,
                        'approach': transformation_rule,
                        'reasoning': reasoning,
                        'confidence': float(confidence) if confidence.replace('.', '').isdigit() else 0.5
                    }
                else:
                    # Failed - record the approach that didn't work
                    failed_approaches.append(f"{transformation_rule} -> Wrong output")
                    return {
                        'success': False,
                        'approach': transformation_rule,
                        'error': 'output_mismatch',
                        'predicted_shape': predicted_output.shape,
                        'expected_shape': expected_output.shape
                    }
            else:
                failed_approaches.append(f"{transformation_rule} -> Could not parse output")
                return {
                    'success': False,
                    'approach': transformation_rule,
                    'error': 'parse_failure'
                }
                
        except Exception as e:
            failed_approaches.append(f"Exception: {str(e)[:100]}")
            return {
                'success': False,
                'approach': 'unknown',
                'error': f'exception: {str(e)[:100]}'
            }
    
    def refine_hypothesis(self, hypothesis: Dict[str, Any], 
                         training_pairs: List[Tuple[np.ndarray, np.ndarray]],
                         validation_failures: List[str]) -> Dict[str, Any]:
        """
        Refine a hypothesis that failed validation
        """
        
        training_context = self._format_training_context(training_pairs)
        failure_context = "\n".join([f"- {failure}" for failure in validation_failures[-5:]])
        
        refinement_prompt = f"""The following hypothesis failed validation. Analyze the failures and create a better hypothesis.

ORIGINAL HYPOTHESIS: {hypothesis['description']}
CATEGORY: {hypothesis['category']}
ORIGINAL EVIDENCE: {hypothesis.get('evidence', 'None')}

TRAINING EXAMPLES:
{training_context}

VALIDATION FAILURES:
{failure_context}

Create a refined hypothesis that:
1. Addresses the validation failures
2. More accurately captures the transformation pattern
3. Stays within the {hypothesis['category']} category
4. Is more specific and testable

Return as JSON:
{{
    "description": "Refined hypothesis description",
    "category": "{hypothesis['category']}",
    "confidence": 0.7,
    "evidence": "Evidence from examples supporting refined hypothesis",
    "refinement_notes": "What was changed and why"
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Refine ARC hypotheses based on validation failures. Be specific and accurate."},
                    {"role": "user", "content": refinement_prompt}
                ],
                temperature=0.2,
                max_tokens=600
            )
            
            content = response.choices[0].message.content
            
            # Try to parse JSON response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                refined_data = json.loads(json_match.group())
                
                # Update original hypothesis with refined data
                refined_hypothesis = {**hypothesis}
                refined_hypothesis.update(refined_data)
                refined_hypothesis['source'] = f"refined_{hypothesis.get('source', 'unknown')}"
                refined_hypothesis['original_description'] = hypothesis['description']
                
                return refined_hypothesis
            else:
                # If JSON parsing fails, create basic refinement
                return {
                    **hypothesis,
                    'description': f"Refined: {hypothesis['description']} (addressing validation failures)",
                    'source': f"refined_{hypothesis.get('source', 'unknown')}",
                    'refinement_notes': "Basic refinement due to parsing failure"
                }
                
        except Exception as e:
            return {
                **hypothesis,
                'description': f"Refinement failed: {hypothesis['description']}",
                'refinement_error': str(e)[:200]
            }
    
    def _format_training_context(self, training_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> str:
        """Format training pairs for AI context"""
        context = ""
        for i, (inp, out) in enumerate(training_pairs):
            context += f"\nExample {i+1}:\n"
            context += f"Input:\n{self._grid_to_string(inp)}\n"
            context += f"Output:\n{self._grid_to_string(out)}\n"
        return context
    
    def _grid_to_string(self, grid: np.ndarray) -> str:
        """Convert numpy grid to readable string"""
        return "\n".join([" ".join([str(cell) for cell in row]) for row in grid])
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a section from AI response"""
        try:
            start = text.find(section_name)
            if start == -1:
                return ""
            
            start += len(section_name)
            next_section = text.find("\n" + section_name.split(":")[0], start)
            if next_section == -1:
                next_section = text.find("\nOUTPUT_GRID:", start)
                if next_section == -1:
                    next_section = text.find("\nCONFIDENCE:", start)
                    if next_section == -1:
                        next_section = len(text)
            
            return text[start:next_section].strip()
        except:
            return ""
    
    def _parse_grid_from_text(self, grid_text: str) -> Optional[np.ndarray]:
        """Parse grid from AI response text"""
        try:
            # Try to find JSON array pattern
            json_match = re.search(r'\[\s*\[.*?\]\s*\]', grid_text, re.DOTALL)
            if json_match:
                grid_data = json.loads(json_match.group())
                return np.array(grid_data)
            
            # Try to parse space/newline separated format
            lines = [line.strip() for line in grid_text.split('\n') if line.strip()]
            if lines:
                grid_data = []
                for line in lines:
                    if any(c.isdigit() for c in line):
                        row = [int(c) for c in line.split() if c.isdigit()]
                        if row:
                            grid_data.append(row)
                
                if grid_data and all(len(row) == len(grid_data[0]) for row in grid_data):
                    return np.array(grid_data)
            
            return None
            
        except Exception:
            return None

class MultiLoopValidator:
    """Handles multiple validation loops to iteratively improve hypotheses"""
    
    def __init__(self, solution_validator: SolutionValidator):
        self.validator = solution_validator
        self.loop_results = {}
    
    def run_validation_loops(self, hypotheses: List[Dict[str, Any]], 
                           training_pairs: List[Tuple[np.ndarray, np.ndarray]],
                           task_name: str, max_loops: int = 2) -> List[Dict[str, Any]]:
        """
        Run multiple validation loops to improve hypotheses
        """
        current_hypotheses = hypotheses.copy()
        loop_history = []
        
        for loop_num in range(max_loops):
            print(f"   Validation loop {loop_num + 1}/{max_loops}...")
            
            validated_hypotheses = []
            refinement_needed = []
            
            # Validate each hypothesis
            for hyp in current_hypotheses:
                validated_hyp = self.validator.validate_hypothesis_by_solving(
                    hyp, training_pairs, task_name
                )
                
                if validated_hyp.get('validation_passed', False):
                    validated_hypotheses.append(validated_hyp)
                elif validated_hyp.get('validation_score', 0) < 0.8:
                    refinement_needed.append(validated_hyp)
                else:
                    validated_hypotheses.append(validated_hyp)
            
            # Refine hypotheses that failed validation
            refined_hypotheses = []
            for hyp in refinement_needed[:3]:  # Only refine top 3 failed hypotheses
                validation_failures = self.validator.validation_attempts.get(
                    f"{task_name}_{hyp.get('description', '')[:50]}", {}
                ).get('failed_approaches', [])
                
                refined_hyp = self.validator.refine_hypothesis(
                    hyp, training_pairs, validation_failures
                )
                refined_hypotheses.append(refined_hyp)
            
            # Update current hypotheses for next loop
            current_hypotheses = validated_hypotheses + refined_hypotheses
            
            # Record loop results
            loop_result = {
                'loop_number': loop_num + 1,
                'validated_count': len(validated_hypotheses),
                'refined_count': len(refined_hypotheses),
                'total_hypotheses': len(current_hypotheses)
            }
            loop_history.append(loop_result)
            
            # Check if we have enough validated hypotheses
            validated_count = sum(1 for h in current_hypotheses if h.get('validation_passed', False))
            if validated_count >= 3:  # Stop if we have 3+ validated hypotheses
                break
        
        # Store results for analysis
        self.loop_results[task_name] = {
            'loops_run': len(loop_history),
            'loop_history': loop_history,
            'final_hypothesis_count': len(current_hypotheses),
            'final_validated_count': sum(1 for h in current_hypotheses if h.get('validation_passed', False))
        }
        
        # Sort by validation score and return
        current_hypotheses.sort(key=lambda x: x.get('validation_score', 0), reverse=True)
        
        return current_hypotheses