from tqdm import tqdm
import numpy as np
from pathlib import Path
from solomonoff import calculate_solomonoff_score
from utils.visualisation import *
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dsl import objects
from ai_hypotheses_generator import AIHypothesisGenerator
from validation_tests import SolutionValidator, MultiLoopValidator
import traceback
from typing import List, Dict, Any, Tuple
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import deque
import random

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class ImprovedHybridRuleEngine:
    def __init__(self, data_folder, output_folder, openai_api_key=None, 
                 max_concurrent_requests: int = 2, rpm_limit: int = 150,
                 enable_validation: bool = True):
        """
        Improved hybrid engine with validation layer
        """
        self.data_folder = Path(data_folder)
        self.output_folder = Path(output_folder)
        self.task_data = {}
        self.category_scores = {}
        self.hypothesis_generator = AIHypothesisGenerator(openai_api_key)
        
        # Validation system
        self.enable_validation = enable_validation
        if enable_validation and openai_api_key:
            self.solution_validator = SolutionValidator(self.hypothesis_generator.client)
            self.multi_loop_validator = MultiLoopValidator(self.solution_validator)
        else:
            self.solution_validator = None
            self.multi_loop_validator = None
        
        # Categories from your definition
        self.categories = [
            "Colour", "Commonsense", "Geometry", 
            "Movement", "Number", "Object"
        ]
        
        # Improved rate limiting
        self.max_concurrent_requests = max_concurrent_requests
        self.rpm_limit = rpm_limit
        self._request_lock = threading.Lock()
        self._recent_call_timestamps = deque()
        self._concurrent_semaphore = threading.BoundedSemaphore(value=self.max_concurrent_requests)
        
        # Performance tracking
        self.ai_calls_made = 0
        self.total_hypotheses_generated = 0
        self.validation_calls_made = 0
        self.validation_improvements = 0
        
        # Problem interpretations storage
        self.problem_interpretations = {}
        
        # Batch processing for efficiency
        self.batch_size = 4
        self.enable_batching = True

    def load_tasks(self):
        """Load all tasks from the data folder with minimal console output"""
        tasks = list(self.data_folder.glob("*.json"))
        for task_path in tasks:
            with open(task_path) as f:
                task = json.load(f)
            task_name = task_path.name
            self.task_data[task_name] = task
        print(f"✅ Loaded {len(self.task_data)} tasks")

    def _wait_for_rate_slot(self):
        """Improved rate limiting"""
        with self._request_lock:
            now = time.monotonic()
            while self._recent_call_timestamps and now - self._recent_call_timestamps[0] > 60.0:
                self._recent_call_timestamps.popleft()

            if len(self._recent_call_timestamps) < self.rpm_limit:
                self._recent_call_timestamps.append(now)
                return
            else:
                earliest = self._recent_call_timestamps[0]
                sleep_for = 60.0 - (now - earliest) + 0.01
        time.sleep(sleep_for)
        return self._wait_for_rate_slot()

    def generate_contextual_interpretation(self, hypotheses: List[Dict[str, Any]], 
                                         task_name: str, task: Dict[str, Any]) -> str:
        """
        Generate a human-readable interpretation using task context and hypotheses
        Enhanced to consider validation results
        """
        if not hypotheses:
            return "Unable to analyze - no hypotheses generated."
        
        # Prioritize validated hypotheses
        validated_hyps = [h for h in hypotheses if h.get('validation_passed', False)]
        if validated_hyps:
            sorted_hyps = sorted(validated_hyps, key=lambda x: (-x.get('validation_score', 0), -x.get('confidence', 0)))
        else:
            sorted_hyps = sorted(hypotheses, key=lambda x: (-x.get('pass_rate', 0), -x.get('confidence', 0)))
        
        # Get task context
        num_examples = len(task.get("train", []))
        training_pairs = [(np.array(pair["input"]), np.array(pair["output"])) for pair in task.get("train", [])]
        
        # Create concise task summary
        task_summary = self._create_task_context_summary(training_pairs)
        
        # Prepare best hypotheses for analysis - prioritize validated ones
        top_hypotheses = []
        for i, hyp in enumerate(sorted_hyps[:5]):
            validation_status = "✓ VALIDATED" if hyp.get('validation_passed') else f"Pass: {hyp.get('pass_rate', 0):.0%}"
            
            hyp_data = {
                'rank': i + 1,
                'category': hyp.get('category', 'Unknown'),
                'description': hyp.get('description', ''),
                'confidence': hyp.get('confidence', 0),
                'pass_rate': hyp.get('pass_rate', 0),
                'validation_score': hyp.get('validation_score', 0),
                'validation_status': validation_status,
                'evidence': hyp.get('evidence', ''),
            }
            top_hypotheses.append(hyp_data)
        
        # Create enhanced prompt for interpretation
        prompt = f"""Analyze this ARC puzzle and provide a clear, human-readable explanation:

TASK: {task_name}
{task_summary}

TOP HYPOTHESES (validated and performance-ranked):
"""
        
        for hyp in top_hypotheses:
            prompt += f"""
{hyp['rank']}. [{hyp['category']}] - {hyp['validation_status']}
   Rule: {hyp['description']}
   Evidence: {hyp['evidence']}"""
        
        prompt += f"""

Focus on the BEST validated hypothesis. Provide a concise analysis:
1. **What this puzzle does** (2-3 sentences max, be specific about the transformation)
2. **Key pattern** (1 sentence describing the core rule)
3. **Category** ({', '.join(self.categories)})
4. **Confidence** (High/Medium/Low based on validation results)

Be clear and specific. Focus on what actually changes from input to output."""

        try:
            response = self.hypothesis_generator.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Provide clear, accurate explanations of ARC puzzle transformations based on validated hypotheses."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.2
            )
            
            interpretation = response.choices[0].message.content
            
            # Add validation info
            has_validated = any(h.get('validation_passed') for h in hypotheses)
            validation_info = f" (Analysis includes {len(validated_hyps)} validated hypotheses)" if has_validated else " (No validated hypotheses)"
            
            return f"🔍 **{task_name}**{validation_info}\n{interpretation}"
            
        except Exception as e:
            print(f"⚠️ Interpretation generation failed: {str(e)[:50]}")
            
            # Enhanced fallback using validation results
            if top_hypotheses:
                best_hyp = top_hypotheses[0]
                validation_note = " (VALIDATED)" if best_hyp.get('validation_status', '').startswith('✓') else ""
                
                return f"""🔍 **{task_name}**{validation_note}

**What this puzzle does:** {best_hyp['description']}
**Key pattern:** {best_hyp['evidence']}
**Category:** {best_hyp['category']}
**Confidence:** {'High' if best_hyp.get('validation_score', 0) >= 0.8 else 'Medium' if best_hyp.get('validation_score', 0) >= 0.6 else 'Low'}
*(Analysis from best-performing hypothesis)*"""
            else:
                return f"🔍 **{task_name}**: Unable to generate interpretation."

    def _create_task_context_summary(self, training_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> str:
        """Create concise task context for interpretations"""
        if not training_pairs:
            return "No training examples available."
        
        summary_parts = []
        
        # Basic info
        summary_parts.append(f"**Examples:** {len(training_pairs)} training pairs")
        
        # Shape analysis
        shapes = [(inp.shape, out.shape) for inp, out in training_pairs]
        if len(set(shapes)) == 1:
            inp_shape, out_shape = shapes[0]
            if inp_shape == out_shape:
                summary_parts.append(f"**Grid size:** {inp_shape[0]}×{inp_shape[1]} (unchanged)")
            else:
                summary_parts.append(f"**Grid size:** {inp_shape[0]}×{inp_shape[1]} → {out_shape[0]}×{out_shape[1]}")
        else:
            summary_parts.append("**Grid size:** varies between examples")
        
        # Color analysis
        all_colors = set()
        for inp, out in training_pairs:
            all_colors.update(inp.flat)
            all_colors.update(out.flat)
        
        summary_parts.append(f"**Colors used:** {len(all_colors)} different colors")
        
        return " | ".join(summary_parts)

    def generate_and_evaluate_hypotheses_batch(self, task: Dict[str, Any], task_name: str = None) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Generate, validate, and evaluate hypotheses with enhanced validation layer"""
        if "train" not in task or not task["train"]:
            return [], {category: 0.0 for category in self.categories}

        # Prepare ALL training pairs for context
        training_pairs = [(np.array(pair["input"]), np.array(pair["output"])) for pair in task["train"]]
        
        # Step 1: Generate initial hypotheses
        acquired = self._concurrent_semaphore.acquire(timeout=30)
        hypotheses = []
        
        try:
            if acquired:
                self._wait_for_rate_slot()
            
            # Use improved generator with full training context
            hypotheses = self.hypothesis_generator.generate_smart_hypotheses(
                training_pairs[0][0], training_pairs[0][1], 
                num_hypotheses=10,  # Slightly fewer initial hypotheses since validation will improve them
                training_pairs=training_pairs
            )
            
            self.ai_calls_made += 1
            self.total_hypotheses_generated += len(hypotheses)
            
        except Exception as e:
            print(f"⚠️ Hypothesis generation failed: {str(e)[:50]}")
            hypotheses = []
            
        finally:
            if acquired:
                self._concurrent_semaphore.release()

        if not hypotheses:
            return [], {category: 0.0 for category in self.categories}

        # Step 2: VALIDATION LAYER - This is the key addition
        if self.enable_validation and self.multi_loop_validator and len(training_pairs) >= 2:
            print(f"   🔬 Running validation loops for {task_name}...")
            
            try:
                validated_hypotheses = self.multi_loop_validator.run_validation_loops(
                    hypotheses, training_pairs, task_name, max_loops=2
                )
                
                # Count validation improvements
                validated_count = sum(1 for h in validated_hypotheses if h.get('validation_passed', False))
                refined_count = sum(1 for h in validated_hypotheses if h.get('source', '').startswith('refined_'))
                
                self.validation_calls_made += len(validated_hypotheses) * 2  # Rough estimate
                self.validation_improvements += refined_count
                
                print(f"   ✓ Validation complete: {validated_count} validated, {refined_count} refined")
                
                hypotheses = validated_hypotheses
                
            except Exception as e:
                print(f"   ⚠️ Validation failed: {str(e)[:50]}")
                # Continue with original hypotheses if validation fails
        
        # Step 3: Traditional testing on ALL training pairs (unchanged)
        tested_hypotheses = []
        
        for hypothesis in hypotheses:
            try:
                test_results = []
                
                # Test on all training pairs
                for pair in task["train"]:
                    inp = np.array(pair["input"])
                    out = np.array(pair["output"])
                    
                    # Get objects using DSL
                    try:
                        inp_objs = objects(tuple(tuple(row) for row in pair["input"]), True, True, True)
                        out_objs = objects(tuple(tuple(row) for row in pair["output"]), True, True, True)
                    except:
                        inp_objs = None
                        out_objs = None
                    
                    # Execute test
                    result = self.execute_hypothesis_test(hypothesis['test_code'], inp, out, inp_objs, out_objs)
                    test_results.append(result)

                # Calculate scores using Solomonoff method
                complexity = self.hypothesis_generator.rule_complexity(hypothesis['description'])
                prior = hypothesis.get('prior_probability', 0.1)
                
                solomonoff_score = calculate_solomonoff_score(test_results, prior, complexity)
                
                hypothesis_result = {
                    **hypothesis,
                    'test_results': test_results,
                    'complexity': complexity,
                    'solomonoff_score': solomonoff_score,
                    'passed_all': all(test_results),
                    'pass_rate': sum(test_results) / len(test_results) if test_results else 0
                }
                
                tested_hypotheses.append(hypothesis_result)
                
            except Exception as e:
                # Silently continue on hypothesis test failures
                continue

        # Step 4: Generate enhanced interpretation considering validation
        if task_name:
            interpretation = self.generate_contextual_interpretation(tested_hypotheses, task_name, task)
            self.problem_interpretations[task_name] = {
                'interpretation': interpretation,
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'num_hypotheses': len(tested_hypotheses),
                'best_pass_rate': max([h.get('pass_rate', 0) for h in tested_hypotheses]) if tested_hypotheses else 0,
                'best_category': max(tested_hypotheses, key=lambda x: x.get('pass_rate', 0)).get('category', 'Unknown') if tested_hypotheses else 'Unknown',
                'validation_enabled': self.enable_validation,
                'validated_count': sum(1 for h in tested_hypotheses if h.get('validation_passed', False)),
                'best_validation_score': max([h.get('validation_score', 0) for h in tested_hypotheses]) if tested_hypotheses else 0
            }

        # Step 5: Enhanced category scoring with validation weights
        category_scores = {category: 0.0 for category in self.categories}

        for hypothesis in tested_hypotheses:
            category = hypothesis['category']
            if category in category_scores:
                # Enhanced scoring: weight by validation results, performance, and confidence
                pass_rate = hypothesis.get('pass_rate', 0)
                confidence = hypothesis.get('confidence', 0.5)
                validation_score = hypothesis.get('validation_score', 0)
                is_validated = hypothesis.get('validation_passed', False)
                
                # Validation bonus: validated hypotheses get significant boost
                validation_multiplier = 3.0 if is_validated else 1.0
                
                if hypothesis['passed_all']:
                    # Perfect hypotheses get exponentially higher scores
                    contribution = 25.0 * confidence * confidence * validation_multiplier
                elif pass_rate >= 0.8:
                    # High-performing hypotheses
                    contribution = 15.0 * pass_rate * confidence * validation_multiplier
                elif pass_rate >= 0.5:
                    # Moderate performance
                    contribution = 8.0 * pass_rate * confidence * validation_multiplier
                else:
                    # Poor performance gets minimal contribution
                    contribution = 2.0 * pass_rate * confidence
                
                # Additional boost for high validation scores
                if validation_score >= 0.8:
                    contribution *= 1.5
                
                category_scores[category] += contribution

        return tested_hypotheses, category_scores

    def execute_hypothesis_test(self, test_code: str, input_grid: np.ndarray, 
                              output_grid: np.ndarray, input_objects=None, 
                              output_objects=None) -> bool:
        """Robust test execution"""
        try:
            namespace = {
                'np': np,
                'input_grid': input_grid,
                'output_grid': output_grid,
                'input_objects': input_objects,
                'output_objects': output_objects,
                'objects': objects,
                'len': len, 'set': set, 'any': any, 'all': all,
                'max': max, 'min': min, 'abs': abs
            }
            
            exec(test_code, namespace)
            result = namespace['test_hypothesis'](input_grid, output_grid, input_objects, output_objects)
            return bool(result)
            
        except Exception:
            return False

    def save_problem_interpretations(self):
        """Save interpretations with validation information"""
        interpretations_path = self.output_folder / "problem_interpretations.json"
        
        # Detailed format with validation info
        detailed_interpretations = {}
        summary_interpretations = {}
        
        for task_name, data in self.problem_interpretations.items():
            # Enhanced detailed version
            detailed_interpretations[task_name] = data
            
            # Enhanced summary version
            clean_interpretation = data['interpretation'].replace('**', '').replace('🔍', '').strip()
            
            # Enhanced confidence calculation
            validation_score = data.get('best_validation_score', 0)
            pass_rate = data.get('best_pass_rate', 0)
            
            if validation_score >= 0.8:
                confidence_level = "High"
            elif validation_score >= 0.6 or pass_rate >= 0.8:
                confidence_level = "Medium"
            else:
                confidence_level = "Low"
            
            # Validation status
            validated_count = data.get('validated_count', 0)
            validation_status = f"{validated_count} validated" if validated_count > 0 else "No validation"
            
            summary_interpretations[task_name] = {
                'what_it_does': clean_interpretation,
                'category': data.get('best_category', 'Unknown'),
                'confidence': confidence_level,
                'success_rate': f"{data.get('best_pass_rate', 0)*100:.0f}%",
                'validation_status': validation_status,
                'validation_score': f"{validation_score*100:.0f}%" if validation_score > 0 else "N/A",
                'generated_at': data.get('generated_at', ''),
                'num_hypotheses_analyzed': data.get('num_hypotheses', 0)
            }
        
        # Save detailed version
        with open(interpretations_path, "w") as f:
            json.dump(detailed_interpretations, f, indent=2)
        
        # Save enhanced summary
        summary_path = self.output_folder / "interpretations_summary.json"
        summary_data = {
            "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_problems": len(summary_interpretations),
            "validation_enabled": self.enable_validation,
            "validation_stats": {
                "total_validation_calls": self.validation_calls_made,
                "total_improvements": self.validation_improvements,
                "tasks_with_validation": sum(1 for s in summary_interpretations.values() 
                                           if s.get('validation_status', '').startswith(('1', '2', '3', '4', '5')))
            },
            "problems": summary_interpretations
        }
        
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"💾 Enhanced interpretations saved: {interpretations_path.name} + {summary_path.name}")
        if self.enable_validation:
            print(f"🔬 Validation stats: {self.validation_calls_made} calls, {self.validation_improvements} improvements")
        
        return interpretations_path

    def manual_categorize(self):
        """Manual categorization interface (same as original)"""
        import matplotlib.pyplot as plt

        self.output_folder.mkdir(parents=True, exist_ok=True)
        manual_path = Path("manual_categorization.json")

        tasks = list(self.data_folder.glob("*.json"))
        manual_results = {}

        print("\n📝 Manual Categorization Mode")
        print("Categories:")
        for i, cat in enumerate(self.categories):
            print(f"{i}: {cat}")

        plt.ion()

        for idx, task_path in enumerate(tasks):
            task_name = task_path.name
            with open(task_path) as f:
                task = json.load(f)

            pairs = [(np.array(pair["input"]), np.array(pair["output"])) for pair in task["train"]]
            fig = compare_multiple_pairs(pairs, task_id=task_name)
            plt.pause(0.001)

            while True:
                try:
                    inp = input(f"\nTask {idx+1}/{len(tasks)}: {task_name}\nEnter category number (or 's' to skip): ").strip()
                    if inp.lower() == 's':
                        print(f"⏭️ Skipped {task_name}")
                        break
                    elif inp.isdigit() and 0 <= int(inp) < len(self.categories):
                        manual_results[task_name] = self.categories[int(inp)]
                        print(f"✔️ Saved: {task_name} → {self.categories[int(inp)]}")
                        break
                    else:
                        print(f"Invalid input. Please enter 0-{len(self.categories)-1} or 's' to skip.")
                except KeyboardInterrupt:
                    print("\nExiting manual categorization.")
                    plt.close("all")
                    return

            plt.close("all")

        with open(manual_path, "w") as f:
            json.dump(manual_results, f, indent=2)

        print(f"✅ Manual categorization saved to: {manual_path}")

    def run(self, save_results=True, enable_validation=None):
        """Main run method with validation integration"""
        if enable_validation is not None:
            self.enable_validation = enable_validation
            
        start_time = time.time()
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Load manual results for accuracy measurement
        manual_path = Path("manual_categorization.json")
        manual_results = {}
        if manual_path.exists():
            with open(manual_path) as f:
                manual_results = json.load(f)

        # Clear existing results
        scores_path = self.output_folder / "evaluated_scores.json"
        hypotheses_path = self.output_folder / "generated_hypotheses.json"
        
        if scores_path.exists():
            scores_path.unlink()

        tasks = list(self.data_folder.glob("*.json"))
        total_tasks = len(tasks)
        all_results = {}
        all_hypotheses = {}

        validation_status = "WITH VALIDATION" if self.enable_validation else "WITHOUT VALIDATION"
        print(f"🚀 Processing {total_tasks} tasks {validation_status}...")

        def process_task_improved(task_path_idx):
            """Process individual task with validation integration"""
            try:
                idx, task_path = task_path_idx
                start = time.time()

                with open(task_path) as f:
                    task = json.load(f)

                task_name = task_path.name
                self.task_data[task_name] = task

                # Generate, validate, and evaluate hypotheses
                hypotheses, category_scores = self.generate_and_evaluate_hypotheses_batch(task, task_name)
                
                if hypotheses is None or category_scores is None:
                    hypotheses = []
                    category_scores = {category: 0.0 for category in self.categories}
                
                # Store hypotheses
                all_hypotheses[task_name] = hypotheses

                # Enhanced score normalization considering validation
                scores = np.array(list(category_scores.values()))
                if np.sum(np.abs(scores)) > 0:
                    # Use softmax but with temperature to preserve strong signals
                    temperature = 0.4  # Slightly lower to emphasize validated results
                    normalized_scores = softmax(scores / temperature)
                else:
                    normalized_scores = np.ones(len(scores)) / len(scores)

                normalized_category_scores = {
                    category: normalized_scores[idx] for idx, category in enumerate(self.categories)
                }

                best_category = max(normalized_category_scores, key=normalized_category_scores.get)
                
                # Prepare task results with validation info
                task['predicted_scores'] = normalized_category_scores
                task['predicted_categories'] = [best_category] * len(task.get("train", []))
                task['generated_hypotheses'] = hypotheses
                task['validation_enabled'] = self.enable_validation
                
                if self.enable_validation:
                    task['validation_summary'] = {
                        'validated_hypotheses': sum(1 for h in hypotheses if h.get('validation_passed', False)),
                        'total_hypotheses': len(hypotheses),
                        'best_validation_score': max([h.get('validation_score', 0) for h in hypotheses]) if hypotheses else 0
                    }
                
                if manual_results and task_name in manual_results:
                    task['expected_category'] = manual_results[task_name]

                # Save individual task file
                if save_results:
                    output_path = self.output_folder / f"{task_path.stem}_evaluated.json"
                    with open(output_path, "w") as out_f:
                        json.dump(task, out_f, indent=2, default=str)

                expected_category = manual_results.get(task_name) if manual_results else None
                
                return (idx, task_name, best_category, expected_category, normalized_category_scores)
                
            except Exception as e:
                print(f"⚠️ Task {task_path.name} failed: {str(e)[:30]}...")
                return (
                    idx, 
                    task_path.name, 
                    "Commonsense",
                    None,
                    {category: 1.0/len(self.categories) for category in self.categories}
                )

        # Process tasks with progress tracking
        correct_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
            futures = {executor.submit(process_task_improved, item): item for item in enumerate(tasks)}
            
            desc = "🔬 AI+Validation" if self.enable_validation else "🤖 AI Analysis"
            with tqdm(total=len(tasks), desc=desc, unit="task", 
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
                
                for future in as_completed(futures):
                    try:
                        idx, task_name, best_cat, expected_cat, norm_scores = future.result()
                        all_results[task_name] = {
                            "predicted_category": best_cat,
                            "expected_category": expected_cat,
                            "scores": norm_scores
                        }
                        
                        if manual_results and expected_cat and expected_cat == best_cat:
                            correct_count += 1
                            
                    except Exception as e:
                        print(f"⚠️ Processing error: {e}")
                        
                    pbar.update(1)

        end_time = time.time()
        
        # Save results
        if save_results:
            with open(scores_path, "w") as f:
                json.dump(all_results, f, indent=2)
            
            with open(hypotheses_path, "w") as f:
                json.dump(all_hypotheses, f, indent=2, default=str)
            
            self.save_problem_interpretations()

        # Enhanced performance summary
        duration = end_time - start_time
        print(f"\n✅ Analysis complete in {duration:.1f}s")
        print(f"🤖 AI calls: {self.ai_calls_made} | Hypotheses: {self.total_hypotheses_generated}")
        
        if self.enable_validation:
            print(f"🔬 Validation calls: {self.validation_calls_made} | Improvements: {self.validation_improvements}")
        
        print(f"📊 Avg: {self.total_hypotheses_generated / len(tasks):.1f} hypotheses/task")

        if manual_results:
            accuracy = correct_count / len(manual_results) * 100
            print(f"🎯 Accuracy: {accuracy:.1f}% ({correct_count}/{len(manual_results)})")

        return all_results

    def View(self, task_name=None, show_all=False, max_tasks=None):
        """Enhanced visualization with validation information"""
        if not self.task_data:
            print("❌ No tasks loaded. Run load_tasks() first.")
            return

        task_names = list(self.task_data.keys())

        # Handle input types
        if isinstance(task_name, int):
            if 0 <= task_name < len(task_names):
                task_names = [task_names[task_name]]
            else:
                print(f"Invalid index: {task_name}. Must be 0-{len(task_names)-1}.")
                return
        elif isinstance(task_name, str):
            task_names = [task_name]
        elif task_name is None:
            if show_all:
                if max_tasks:
                    task_names = task_names[:max_tasks]
            else:
                task_names = task_names[:1]

        print(f"\n📊 Analyzing {len(task_names)} task(s)")
        
        for i, task_name in enumerate(task_names):
            if i > 0 and show_all and len(task_names) > 5:
                response = input(f"\n⏸️ Continue to task {i+1}/{len(task_names)}? (Enter/q): ")
                if response.lower() == 'q':
                    print("Exiting view mode.")
                    break
            
            print(f"\n🔍 TASK {i+1}/{len(task_names)}: {task_name}")
            print("=" * 50)

            # Load task data
            evaluated_path = self.output_folder / task_name.replace(".json", "_evaluated.json")
            if evaluated_path.exists():
                with open(evaluated_path) as f:
                    task = json.load(f)
            else:
                task = self.task_data.get(task_name)

            if task is None:
                print(f"❌ Task '{task_name}' not found.")
                continue

            # Show prediction vs expected
            predicted_categories = task.get("predicted_categories", [])
            expected_category = task.get("expected_category", None)
            predicted = predicted_categories[0] if predicted_categories else "Unknown"
            
            match_symbol = "✅" if predicted == expected_category else "❌"
            print(f"**Prediction:** {predicted} | **Expected:** {expected_category} {match_symbol}")

            # Show validation summary
            validation_summary = task.get('validation_summary', {})
            if validation_summary:
                validated = validation_summary.get('validated_hypotheses', 0)
                total = validation_summary.get('total_hypotheses', 0)
                best_val_score = validation_summary.get('best_validation_score', 0)
                print(f"**Validation:** {validated}/{total} hypotheses validated (best: {best_val_score:.2f})")

            # Show interpretation
            if task_name in self.problem_interpretations:
                interpretation = self.problem_interpretations[task_name]['interpretation']
                print(f"\n{interpretation}")
            
            # Show category scores
            predicted_scores = task.get("predicted_scores", {})
            if predicted_scores:
                print(f"\n**Top Categories:**")
                sorted_scores = sorted(predicted_scores.items(), key=lambda x: -x[1])[:3]
                for cat, score in sorted_scores:
                    print(f"  {cat}: {score:.2f}")

            # Enhanced hypothesis summary with validation info
            hypotheses = task.get('generated_hypotheses', [])
            if hypotheses:
                # Sort by validation first, then by pass rate
                sorted_hyps = sorted(hypotheses, key=lambda x: (
                    -float(x.get('validation_passed', False)),
                    -x.get('validation_score', 0),
                    -x.get('pass_rate', 0),
                    -x.get('confidence', 0)
                ))
                
                validated_count = sum(1 for h in sorted_hyps if h.get('validation_passed', False))
                passed_all = sum(1 for h in sorted_hyps if h.get('passed_all', False))
                
                print(f"\n**Hypotheses:** {len(hypotheses)} generated | {validated_count} validated | {passed_all} passed all tests")
                
                # Show top hypotheses with validation status
                for j, hyp in enumerate(sorted_hyps[:3]):
                    if hyp.get('validation_passed', False):
                        status = f"🔬 VALIDATED ({hyp.get('validation_score', 0):.2f})"
                    elif hyp.get('passed_all', False):
                        status = "✅ PASSED ALL"
                    else:
                        status = f"❌ {hyp.get('pass_rate', 0):.0%}"
                    
                    refined_note = " (REFINED)" if hyp.get('source', '').startswith('refined_') else ""
                    
                    print(f"  {j+1}. [{hyp['category']}] {status}{refined_note}")
                    print(f"     {hyp['description']}")

            # Visualize
            try:
                pairs = [(np.array(pair["input"]), np.array(pair["output"])) for pair in task["train"]]
                compare_multiple_pairs(pairs, task_id=task_name, 
                                     predicted_categories=predicted_categories,
                                     expected_category=expected_category)
                plt.show()
                
                if show_all and len(task_names) > 1:
                    plt.close('all')
                    
            except Exception as e:
                print(f"⚠️ Visualization error: {str(e)[:50]}")

    def get_performance_summary(self):
        """Enhanced performance summary with validation metrics"""
        scores_path = self.output_folder / "evaluated_scores.json"
        if not scores_path.exists():
            print("❌ No results found. Run the engine first.")
            return
            
        with open(scores_path) as f:
            all_results = json.load(f)
        
        total_with_expected = sum(1 for r in all_results.values() if r.get('expected_category'))
        correct = sum(1 for r in all_results.values() 
                     if r.get('expected_category') and r.get('predicted_category') == r.get('expected_category'))
        
        if total_with_expected == 0:
            print("No manual categorizations found for comparison.")
            return
            
        accuracy = correct / total_with_expected * 100
        
        print(f"\n📊 PERFORMANCE SUMMARY")
        print("=" * 25)
        print(f"Accuracy: {accuracy:.1f}% ({correct}/{total_with_expected})")
        print(f"AI calls: {self.ai_calls_made}")
        print(f"Hypotheses: {self.total_hypotheses_generated}")
        
        if self.enable_validation:
            print(f"Validation calls: {self.validation_calls_made}")
            print(f"Validation improvements: {self.validation_improvements}")
            
        print(f"Interpretations: {len(self.problem_interpretations)}")
        
        # Category performance (unchanged)
        category_perf = {cat: {"correct": 0, "total": 0} for cat in self.categories}
        
        for result in all_results.values():
            expected = result.get('expected_category')
            predicted = result.get('predicted_category')
            
            if expected and expected in category_perf:
                category_perf[expected]["total"] += 1
                if predicted == expected:
                    category_perf[expected]["correct"] += 1
        
        print(f"\n📈 Category Performance:")
        for cat, perf in category_perf.items():
            if perf["total"] > 0:
                cat_acc = perf["correct"] / perf["total"] * 100
                print(f"  {cat:11}: {cat_acc:5.1f}% ({perf['correct']}/{perf['total']})")

    # [Rest of the methods remain unchanged - view_interpretations, analyze_errors, etc.]
    def view_interpretations(self, task_name=None, max_display=10):
        """View problem interpretations with validation information"""
        interpretations_path = self.output_folder / "interpretations_summary.json"
        
        if not interpretations_path.exists():
            print("❌ No interpretations found. Run the engine first.")
            return
        
        with open(interpretations_path) as f:
            data = json.load(f)
        
        interpretations = data.get("problems", {})
        
        if not interpretations:
            print("No interpretations available.")
            return
        
        # Show validation stats if available
        validation_stats = data.get("validation_stats", {})
        if validation_stats:
            print(f"🔬 Validation stats: {validation_stats}")
        
        if task_name:
            # Show specific task with validation info
            if task_name in interpretations:
                interp_data = interpretations[task_name]
                print(f"\n🔍 {task_name}")
                print("=" * 50)
                print(f"**What it does:** {interp_data['what_it_does']}")
                print(f"**Category:** {interp_data['category']}")
                print(f"**Confidence:** {interp_data['confidence']} ({interp_data['success_rate']})")
                
                val_status = interp_data.get('validation_status', 'No validation')
                val_score = interp_data.get('validation_score', 'N/A')
                print(f"**Validation:** {val_status} (score: {val_score})")
            else:
                print(f"❌ No interpretation found for: {task_name}")
        else:
            # Show summary with validation info
            print(f"\n📚 PROBLEM INTERPRETATIONS ({len(interpretations)} tasks)")
            if data.get('validation_enabled'):
                print("🔬 Validation-enhanced analysis")
            print("=" * 60)
            
            displayed = 0
            for task, interp_data in interpretations.items():
                if displayed >= max_display:
                    remaining = len(interpretations) - displayed
                    print(f"\n... and {remaining} more interpretations")
                    print("Use view_interpretations('task_name.json') for specific tasks")
                    break
                
                confidence_icon = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(interp_data['confidence'], "⚪")
                val_status = interp_data.get('validation_status', '')
                val_indicator = "🔬" if val_status and not val_status.startswith('No') else ""
                
                print(f"\n{displayed + 1}. {task} {val_indicator}")
                print(f"   {confidence_icon} {interp_data['category']} | {interp_data['confidence']} confidence")
                
                if val_status and not val_status.startswith('No'):
                    print(f"   🔬 {val_status}")
                
                # Truncate description if too long
                description = interp_data['what_it_does']
                if len(description) > 100:
                    description = description[:97] + "..."
                print(f"   {description}")
                
                displayed += 1

    def analyze_errors(self, show_details=True):
        """Analyze prediction errors with validation context"""
        scores_path = self.output_folder / "evaluated_scores.json"
        if not scores_path.exists():
            print("❌ No results found.")
            return
            
        with open(scores_path) as f:
            all_results = json.load(f)
        
        errors = []
        for task_name, result in all_results.items():
            predicted = result.get('predicted_category')
            expected = result.get('expected_category')
            if expected and predicted != expected:
                errors.append((task_name, predicted, expected, result.get('scores', {})))
        
        if not errors:
            print("🎉 No prediction errors found!")
            return
            
        print(f"\n❌ ERROR ANALYSIS ({len(errors)} errors)")
        if self.enable_validation:
            print("🔬 With validation-enhanced analysis")
        print("=" * 40)
        
        # Group by error pattern (unchanged logic)
        error_patterns = {}
        for task_name, predicted, expected, scores in errors:
            pattern = f"{expected} → {predicted}"
            if pattern not in error_patterns:
                error_patterns[pattern] = []
            error_patterns[pattern].append((task_name, scores))
        
        # Show most common patterns
        sorted_patterns = sorted(error_patterns.items(), key=lambda x: -len(x[1]))
        for pattern, cases in sorted_patterns[:5]:
            print(f"\n**{pattern}:** {len(cases)} cases")
            
            if show_details:
                # Show confidence analysis
                avg_correct_score = np.mean([scores.get(pattern.split(' → ')[0], 0) for _, scores in cases])
                avg_wrong_score = np.mean([scores.get(pattern.split(' → ')[1], 0) for _, scores in cases])
                
                print(f"  Avg score for correct category: {avg_correct_score:.3f}")
                print(f"  Avg score for predicted category: {avg_wrong_score:.3f}")
                
                # Show example tasks
                for task_name, _ in cases[:2]:
                    print(f"    - {task_name}")
                if len(cases) > 2:
                    print(f"    ... and {len(cases) - 2} more")
        
        return errors

# Backwards compatibility class
class HybridRuleEngine(ImprovedHybridRuleEngine):
    """Backwards compatible interface"""
    pass