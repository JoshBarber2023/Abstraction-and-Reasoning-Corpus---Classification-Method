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

class HybridRuleEngine:
    def __init__(self, data_folder, output_folder, openai_api_key=None, 
                 max_concurrent_requests: int = 3, rpm_limit: int = 200):
        """
        Hybrid engine that combines AI hypothesis generation with efficient batch evaluation
        """
        self.data_folder = Path(data_folder)
        self.output_folder = Path(output_folder)
        self.task_data = {}
        self.category_scores = {}
        self.hypothesis_generator = AIHypothesisGenerator(openai_api_key)
        
        # Categories from your definition
        self.categories = [
            "Colour", "Commonsense", "Geometry", 
            "Movement", "Number", "Object"
        ]
        
        # Rate limiting (similar to your original approach)
        self.max_concurrent_requests = max_concurrent_requests
        self.rpm_limit = rpm_limit
        self._request_lock = threading.Lock()
        self._recent_call_timestamps = deque()
        self._concurrent_semaphore = threading.BoundedSemaphore(value=self.max_concurrent_requests)
        
        # Performance tracking
        self.ai_calls_made = 0
        self.total_hypotheses_generated = 0
        
        # Problem interpretations storage
        self.problem_interpretations = {}

    def load_tasks(self):
        """Load all tasks from the data folder"""
        tasks = list(self.data_folder.glob("*.json"))
        for task_path in tasks:
            with open(task_path) as f:
                task = json.load(f)
            task_name = task_path.name
            self.task_data[task_name] = task
        print(f"Loaded {len(self.task_data)} tasks")

    def _wait_for_rate_slot(self):
        """Rate limiting from your original code"""
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

    def generate_problem_interpretation(self, hypotheses: List[Dict[str, Any]], task_name: str) -> str:
        """
        Generate a plain-language interpretation of what the problem is doing
        by sending all hypotheses to ChatGPT for analysis
        """
        if not hypotheses:
            return "Unable to determine what this problem is doing - no hypotheses generated."
        
        # Sort hypotheses by performance (pass rate, then confidence)
        sorted_hyps = sorted(hypotheses, key=lambda x: (-x.get('pass_rate', 0), -x.get('confidence', 0)))
        
        # Prepare hypotheses data for ChatGPT
        hypotheses_summary = []
        for i, hyp in enumerate(sorted_hyps):
            hyp_data = {
                'rank': i + 1,
                'category': hyp.get('category', 'Unknown'),
                'description': hyp.get('description', ''),
                'confidence': hyp.get('confidence', 0),
                'pass_rate': hyp.get('pass_rate', 0),
                'passed_all_training': hyp.get('passed_all', False),
                'evidence': hyp.get('evidence', ''),
                'test_results': hyp.get('test_results', [])
            }
            hypotheses_summary.append(hyp_data)
        
        # Create prompt for ChatGPT
        prompt = f"""I have an ARC (Abstraction and Reasoning Corpus) puzzle task called "{task_name}" and I've generated {len(hypotheses)} different hypotheses about what transformation rule this puzzle follows.

    Please analyze these hypotheses and provide a clear, concise interpretation of what you think this puzzle is actually doing. Focus on:
    1. What is the core transformation or pattern?
    2. What are the key elements being modified?
    3. How confident should we be in this interpretation?

    Here are the hypotheses ranked by performance:

    """
        
        # Add each hypothesis to the prompt
        for hyp in hypotheses_summary:
            success_rate = f"{hyp['pass_rate']*100:.0f}%" if hyp['pass_rate'] > 0 else "0%"
            status = "✓ PASSED ALL" if hyp['passed_all_training'] else f"✗ {success_rate} success"
            
            prompt += f"""
    Hypothesis #{hyp['rank']} [{hyp['category']}] - {status}
    Description: {hyp['description']}
    Confidence: {hyp['confidence']:.2f}
    Evidence: {hyp['evidence']}
    Training Results: {hyp['test_results']}
    ---"""
        
        prompt += f"""

    Based on this analysis, please provide:

    1. **Most Likely Explanation**: What do you think this puzzle is actually doing? (2-3 sentences)
    2. **Key Pattern**: What is the core transformation rule? (1-2 sentences)  
    3. **Confidence Level**: How confident are you in this interpretation? (High/Medium/Low and why)
    4. **Category**: Which category best fits this puzzle? (Colour, Commonsense, Geometry, Movement, Number, Object)

    Please be concise and focus on the most probable explanation based on the hypothesis performance."""

        try:
            # Use the existing hypothesis generator's API connection
            response = self.hypothesis_generator.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing ARC puzzles and understanding transformation patterns. Provide clear, concise interpretations based on the provided hypotheses."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            interpretation = response.choices[0].message.content
            
            # Format the response nicely
            formatted_interpretation = f"🔍 **Problem Analysis for {task_name}:**\n\n"
            formatted_interpretation += interpretation
            formatted_interpretation += f"\n\n**Analysis based on {len(hypotheses)} AI-generated hypotheses**"
            formatted_interpretation += f"\n**Generated at:** {time.strftime('%Y-%m-%d %H:%M:%S')}"
            
            return formatted_interpretation
            
        except Exception as e:
            print(f"Error generating ChatGPT interpretation: {str(e)}")
            
            # Fallback to best hypothesis if ChatGPT fails
            if hypotheses:
                best_hyp = sorted_hyps[0]
                pass_rate = best_hyp.get('pass_rate', 0) * 100
                
                fallback_interpretation = f"🔍 **Problem Analysis for {task_name}:**\n\n"
                fallback_interpretation += f"**Best Hypothesis** (ChatGPT unavailable):\n"
                fallback_interpretation += f"**Description:** {best_hyp.get('description', '')}\n"
                fallback_interpretation += f"**Category:** {best_hyp.get('category', 'Unknown')}\n"
                fallback_interpretation += f"**Success Rate:** {pass_rate:.0f}% on training examples\n"
                fallback_interpretation += f"**Evidence:** {best_hyp.get('evidence', 'N/A')}"
                
                return fallback_interpretation
            else:
                return "Unable to generate interpretation - no hypotheses available and ChatGPT unavailable."

    def generate_and_evaluate_hypotheses(self, task: Dict[str, Any], task_name: str = None) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Generate AI hypotheses and evaluate them like your original approach"""
        if "train" not in task or not task["train"]:
            return [], {category: 0.0 for category in self.categories}

        # Use first training pair for hypothesis generation
        first_pair = task["train"][0]
        input_grid = np.array(first_pair["input"])
        output_grid = np.array(first_pair["output"])
        
        # Generate AI hypotheses with rate limiting and better error handling
        acquired = self._concurrent_semaphore.acquire(timeout=30)
        hypotheses = []  # Initialize with empty list
        try:
            if acquired:
                self._wait_for_rate_slot()
            
            hypotheses = self.hypothesis_generator.generate_smart_hypotheses(
                input_grid, output_grid, num_hypotheses=8
            )
            self.ai_calls_made += 1
            self.total_hypotheses_generated += len(hypotheses)
            
        except Exception as e:
            print(f"Error generating hypotheses: {str(e)[:100]}")
            hypotheses = []  # Use empty list on failure
            
        finally:
            if acquired:
                self._concurrent_semaphore.release()

        if not hypotheses:
            return [], {category: 0.0 for category in self.categories}

        # Test hypotheses on ALL training pairs (like your original)
        tested_hypotheses = []
        
        for hypothesis in hypotheses:
            try:
                test_results = []
                
                # Test on all training pairs
                for pair in task["train"]:
                    inp = np.array(pair["input"])
                    out = np.array(pair["output"])
                    
                    # Get objects using DSL (like your original)
                    try:
                        inp_objs = objects(tuple(tuple(row) for row in pair["input"]), True, True, True)
                        out_objs = objects(tuple(tuple(row) for row in pair["output"]), True, True, True)
                    except:
                        inp_objs = None
                        out_objs = None
                    
                    # Execute test (robust like your original)
                    result = self.execute_hypothesis_test(hypothesis['test_code'], inp, out, inp_objs, out_objs)
                    test_results.append(result)

                # Calculate scores using your Solomonoff method
                complexity = self.hypothesis_generator.rule_complexity(hypothesis['description'])
                prior = hypothesis.get('prior_probability', 0.1)
                
                # Use your exact scoring method
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
                print(f"Error testing hypothesis: {str(e)[:100]}")
                continue

        # Generate problem interpretation
        if task_name:
            interpretation = self.generate_problem_interpretation(tested_hypotheses, task_name)
            self.problem_interpretations[task_name] = {
                'interpretation': interpretation,
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'num_hypotheses': len(tested_hypotheses),
                'best_pass_rate': max([h.get('pass_rate', 0) for h in tested_hypotheses]) if tested_hypotheses else 0,
                'best_category': max(tested_hypotheses, key=lambda x: x.get('pass_rate', 0)).get('category', 'Unknown') if tested_hypotheses else 'Unknown'
            }

        # Calculate category scores (like your original approach)
        category_scores = {category: 0.0 for category in self.categories}

        for hypothesis in tested_hypotheses:
            category = hypothesis['category']
            if category in category_scores:
                # Use POSITIVE scores - higher is better
                if hypothesis['passed_all']:
                    # Perfect hypotheses get high positive scores
                    contribution = 10.0 * hypothesis.get('confidence', 0.5)
                elif hypothesis['pass_rate'] > 0:
                    # Partial matches get moderate scores
                    contribution = hypothesis['pass_rate'] * hypothesis.get('confidence', 0.5)
                else:
                    # Failed hypotheses get zero contribution
                    contribution = 0.0
                
                category_scores[category] += contribution

        # FIX: Return the results!
        return tested_hypotheses, category_scores

    def execute_hypothesis_test(self, test_code: str, input_grid: np.ndarray, 
                              output_grid: np.ndarray, input_objects=None, 
                              output_objects=None) -> bool:
        """Robust test execution like your original"""
        try:
            # Safe execution environment
            namespace = {
                'np': np,
                'input_grid': input_grid,
                'output_grid': output_grid,
                'input_objects': input_objects,
                'output_objects': output_objects,
                'objects': objects,  # Include DSL objects function
                'len': len,
                'set': set,
                'any': any,
                'all': all,
                'max': max,
                'min': min
            }
            
            # Execute test code
            exec(test_code, namespace)
            
            # Call test function
            result = namespace['test_hypothesis'](input_grid, output_grid, input_objects, output_objects)
            return bool(result)
            
        except Exception as e:
            # Fail gracefully
            return False

    def save_problem_interpretations(self):
        """Save all problem interpretations to a JSON file"""
        interpretations_path = self.output_folder / "problem_interpretations.json"
        
        # Convert interpretations to a more readable format for JSON
        json_friendly_interpretations = {}
        for task_name, data in self.problem_interpretations.items():
            json_friendly_interpretations[task_name] = {
                'plain_text_interpretation': data['interpretation'].replace('**', '').replace('🔍', '').replace('\n', ' '),
                'generated_at': data['generated_at'],
                'confidence_metrics': {
                    'num_hypotheses': data['num_hypotheses'],
                    'best_pass_rate': data['best_pass_rate'],
                    'best_category': data['best_category']
                },
                'full_interpretation': data['interpretation']  # Keep formatted version too
            }
        
        with open(interpretations_path, "w") as f:
            json.dump(json_friendly_interpretations, f, indent=2)
        
        print(f"✅ Problem interpretations saved to: {interpretations_path.resolve()}")
        return interpretations_path

    def manual_categorize(self):
        """Manual categorization interface (same as your original)"""
        import matplotlib.pyplot as plt

        self.output_folder.mkdir(parents=True, exist_ok=True)
        manual_path = Path("manual_categorization.json")

        tasks = list(self.data_folder.glob("*.json"))
        manual_results = {}

        print("\nManual Categorization Mode")
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
                        print(f"Invalid input. Please enter a number between 0 and {len(self.categories)-1}, or 's' to skip.")
                except KeyboardInterrupt:
                    print("\nExiting manual categorization.")
                    plt.close("all")
                    return

            plt.close("all")

        with open(manual_path, "w") as f:
            json.dump(manual_results, f, indent=2)

        print(f"\n✅ Manual categorization saved to: {manual_path}")

    def run(self, save_results=True):
        """Main run method with improved accuracy"""
        start_time = time.time()
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Load manual results
        manual_path = Path("manual_categorization.json")
        manual_results = {}
        if manual_path.exists():
            with open(manual_path) as f:
                manual_results = json.load(f)

        scores_path = self.output_folder / "evaluated_scores.json"
        hypotheses_path = self.output_folder / "generated_hypotheses.json"
        
        if scores_path.exists():
            scores_path.unlink()
            print(f"🗑️ Deleted existing {scores_path.name}")

        tasks = list(self.data_folder.glob("*.json"))
        total_tasks = len(tasks)
        all_results = {}
        all_hypotheses = {}
        correct_count = 0
        correct_tasks = []

        def process_task(task_path_idx):
            """Process individual task with AI hypothesis generation"""
            try:
                idx, task_path = task_path_idx
                start = time.time()

                with open(task_path) as f:
                    task = json.load(f)

                task_name = task_path.name
                self.task_data[task_name] = task

                # Generate and evaluate hypotheses (now includes interpretation generation)
                hypotheses, category_scores = self.generate_and_evaluate_hypotheses(task, task_name)
                
                # If generation failed, use empty results
                if hypotheses is None or category_scores is None:
                    hypotheses = []
                    category_scores = {category: 0.0 for category in self.categories}
                
                # Store hypotheses for analysis
                all_hypotheses[task_name] = hypotheses

                # Normalize scores using softmax (like your original)
                scores = np.array(list(category_scores.values()))
                if np.sum(np.abs(scores)) > 0:
                    normalized_scores = softmax(scores)
                else:
                    normalized_scores = np.ones(len(scores)) / len(scores)

                normalized_category_scores = {
                    category: normalized_scores[idx] for idx, category in enumerate(self.categories)
                }

                best_category = max(normalized_category_scores, key=normalized_category_scores.get)
                
                # Add results to task
                task['predicted_scores'] = normalized_category_scores
                task['predicted_categories'] = [best_category] * len(task.get("train", []))
                task['generated_hypotheses'] = hypotheses  # Include hypotheses in task file
                
                if manual_results and task_name in manual_results:
                    task['expected_category'] = manual_results[task_name]

                # Save individual task file
                if save_results:
                    output_path = self.output_folder / f"{task_path.stem}_evaluated.json"
                    with open(output_path, "w") as out_f:
                        json.dump(task, out_f, indent=2, default=str)

                expected_category = manual_results.get(task_name) if manual_results else None
                duration = time.time() - start
                
                return (idx, task_name, best_category, expected_category, normalized_category_scores)
                
            except Exception as e:
                print(f"Error processing task {task_path.name}: {str(e)[:100]}")
                # Return a safe default result instead of None
                return (
                    idx, 
                    task_path.name, 
                    "Commonsense",  # Default category
                    None,  # No expected category
                    {category: 1.0/len(self.categories) for category in self.categories}  # Equal probabilities
                )

        # Process tasks with controlled concurrency
        print(f"🚀 Processing {total_tasks} tasks with AI hypothesis generation...")
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
            futures = {executor.submit(process_task, item): item for item in enumerate(tasks)}
            
            with tqdm(total=len(tasks), desc="AI Analysis", unit="task") as pbar:
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
                            correct_tasks.append((idx, task_name))
                            
                    except Exception as e:
                        print(f"Error processing task: {e}")
                        
                    pbar.update(1)

        end_time = time.time()
        
        # Save results
        if save_results:
            with open(scores_path, "w") as f:
                json.dump(all_results, f, indent=2)
            
            with open(hypotheses_path, "w") as f:
                json.dump(all_hypotheses, f, indent=2, default=str)
            
            # Save problem interpretations
            self.save_problem_interpretations()
            
            print(f"\n✅ Results saved to: {scores_path.resolve()}")
            print(f"✅ Hypotheses saved to: {hypotheses_path.resolve()}")

        # Performance summary
        print(f"\n⚡ Completed in {end_time - start_time:.1f} seconds")
        print(f"🤖 AI calls made: {self.ai_calls_made}")
        print(f"📊 Total hypotheses: {self.total_hypotheses_generated}")
        print(f"📊 Avg hypotheses per task: {self.total_hypotheses_generated / len(tasks):.1f}")
        print(f"🧠 Problem interpretations generated: {len(self.problem_interpretations)}")

        if manual_results:
            accuracy = correct_count / len(manual_results) * 100
            print(f"\n🎯 Accuracy: {accuracy:.2f}% ({correct_count}/{len(manual_results)})")
            
            if correct_tasks:
                print("✅ Sample correct predictions:")
                for idx, task in sorted(correct_tasks)[:10]:  # Show first 10
                    print(f" - [#{idx}] {task}")
                if len(correct_tasks) > 10:
                    print(f" ... and {len(correct_tasks) - 10} more")

    def View(self, task_name=None, show_all=False, max_tasks=None):
        """Enhanced visualization with detailed hypothesis analysis and problem interpretation"""
        if not self.task_data:
            print("No tasks loaded. Run load_tasks() first.")
            return

        task_names = list(self.task_data.keys())

        # Handle input types
        if isinstance(task_name, int):
            if 0 <= task_name < len(task_names):
                task_names = [task_names[task_name]]
            else:
                print(f"Invalid index: {task_name}. Must be between 0 and {len(task_names)-1}.")
                return
        elif isinstance(task_name, str):
            task_names = [task_name]
        elif task_name is None:
            if show_all:
                # Show all tasks
                if max_tasks:
                    task_names = task_names[:max_tasks]
            else:
                task_names = task_names[:1]  # Show first task by default

        print(f"\n📊 Viewing {len(task_names)} task(s)")
        
        for i, task_name in enumerate(task_names):
            if i > 0 and show_all:
                # Add pause between tasks when viewing multiple
                response = input(f"\nPress Enter to continue to task {i+1}/{len(task_names)} (or 'q' to quit): ")
                if response.lower() == 'q':
                    print("Exiting view mode.")
                    break
            
            print(f"\n🔍 ANALYZING TASK {i+1}/{len(task_names)}: {task_name}")
            print("=" * 60)

            # Load task data
            evaluated_path = self.output_folder / task_name.replace(".json", "_evaluated.json")
            if evaluated_path.exists():
                with open(evaluated_path) as f:
                    task = json.load(f)
            else:
                task = self.task_data.get(task_name)

            if task is None:
                print(f"Task '{task_name}' not found.")
                continue

            # Show prediction vs expected
            predicted_categories = task.get("predicted_categories", [])
            expected_category = task.get("expected_category", None)
            predicted = predicted_categories[0] if predicted_categories else "Unknown"
            
            match_symbol = "✅" if predicted == expected_category else "❌"
            print(f"Prediction: {predicted} | Expected: {expected_category} {match_symbol}")
            
            # Show category scores
            predicted_scores = task.get("predicted_scores", {})
            if predicted_scores:
                print(f"\nCategory Confidence Scores:")
                sorted_scores = sorted(predicted_scores.items(), key=lambda x: -x[1])
                for cat, score in sorted_scores:
                    bar = "█" * int(score * 20)  # Visual bar
                    print(f"  {cat:12}: {score:.3f} {bar}")

            # NEW: Show problem interpretation
            if task_name in self.problem_interpretations:
                print(f"\n🤖 WHAT I THINK IS HAPPENING:")
                print("-" * 50)
                interpretation_data = self.problem_interpretations[task_name]
                
                # Show the full interpretation
                full_interpretation = interpretation_data.get('interpretation', 'No interpretation available')
                print(full_interpretation)
                
                # Show confidence metrics
                print(f"\n📊 Analysis Confidence:")
                print(f"  • Generated from {interpretation_data.get('num_hypotheses', 0)} hypotheses")
                print(f"  • Best hypothesis passed {interpretation_data.get('best_pass_rate', 0)*100:.0f}% of training examples")
                print(f"  • Primary category: {interpretation_data.get('best_category', 'Unknown')}")
            else:
                print(f"\n🤖 WHAT I THINK IS HAPPENING:")
                print("-" * 50)
                print("No interpretation available - run the engine first to generate interpretations.")

            # Analyze generated hypotheses in detail
            hypotheses = task.get('generated_hypotheses', [])
            if hypotheses:
                print(f"\n🧠 GENERATED HYPOTHESES ({len(hypotheses)}):")
                print("-" * 50)
                
                # Sort by performance
                sorted_hyps = sorted(hypotheses, key=lambda x: (-x.get('pass_rate', 0), -x.get('confidence', 0)))
                
                for j, hyp in enumerate(sorted_hyps):
                    status = "✅ PASSED ALL" if hyp.get('passed_all', False) else f"❌ {hyp.get('pass_rate', 0):.0%} PASS"
                    confidence_bar = "★" * int(hyp.get('confidence', 0) * 5)
                    
                    print(f"\n{j+1}. [{hyp['category']}] {status}")
                    print(f"   Description: {hyp['description']}")
                    print(f"   Confidence: {hyp.get('confidence', 0):.2f} {confidence_bar}")
                    print(f"   Evidence: {hyp.get('evidence', 'N/A')}")
                    print(f"   Solomonoff Score: {hyp.get('solomonoff_score', 0):.3f}")
                    print(f"   Test Results: {hyp.get('test_results', [])}")
                    
                    # Show why it failed if it did
                    if not hyp.get('passed_all', False):
                        failed_pairs = [k for k, result in enumerate(hyp.get('test_results', [])) if not result]
                        if failed_pairs:
                            print(f"   ❌ Failed on training pairs: {failed_pairs}")
                
                # Category analysis
                by_category = {}
                for hyp in hypotheses:
                    cat = hyp['category']
                    if cat not in by_category:
                        by_category[cat] = []
                    by_category[cat].append(hyp)
                
                print(f"\n📊 HYPOTHESIS BREAKDOWN:")
                print("-" * 30)
                for cat, cat_hyps in by_category.items():
                    passed_all = sum(1 for h in cat_hyps if h.get('passed_all', False))
                    avg_pass_rate = np.mean([h.get('pass_rate', 0) for h in cat_hyps])
                    print(f"{cat:12}: {len(cat_hyps)} hyps | {passed_all} perfect | {avg_pass_rate:.2f} avg pass rate")

            # Visualize the grids with object numbers
            try:
                pairs = [(np.array(pair["input"]), np.array(pair["output"])) for pair in task["train"]]
                
                # Enhanced visualization with object numbers
                compare_multiple_pairs(
                    pairs, 
                    task_id=task_name, 
                    predicted_categories=predicted_categories,
                    expected_category=expected_category,
                    show_object_numbers=True  # Show object numbers on grids
                )

                # Optional: Show a separate legend for object meanings
                if pairs:
                    # Create legend based on all unique values from all pairs
                    all_values = set()
                    for inp, out in pairs:
                        all_values.update(inp.flatten())
                        all_values.update(out.flatten())
                    
                    if len(all_values) <= 15:  # Only show legend if not too many unique values
                        plt.figure(figsize=(4, max(3, len(all_values) * 0.3)))
                        legend_text = "🔢 Object Number Legend:\n\n"
                        for val in sorted(all_values):
                            legend_text += f"   {int(val)} = Object {int(val)}\n"
                        
                        plt.text(0.1, 0.95, legend_text, 
                                fontsize=11, verticalalignment='top',
                                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
                        plt.title(f"Object Reference for {task_name}", fontsize=12, pad=20)
                        plt.axis('off')

                # Show category scores plot
                try:
                    scores_path = self.output_folder / "evaluated_scores.json"
                    if scores_path.exists():
                        with open(scores_path, "r") as f:
                            self.category_scores = json.load(f)
                            
                        score_dict = self.category_scores.get(task_name, {})
                        if score_dict:
                            plot_solomonoff_scores(score_dict)
                            
                except Exception as e:
                    print(f"Could not plot scores: {e}")

                plt.show()
                
                # If viewing multiple tasks, close plots after showing
                if show_all and len(task_names) > 1:
                    plt.close('all')
                    
            except Exception as e:
                print(f"Visualization error: {e}")
                import traceback
                traceback.print_exc()

    def analyze_errors(self):
        """Analyze common error patterns to improve the system"""
        scores_path = self.output_folder / "evaluated_scores.json"
        if not scores_path.exists():
            print("❌ No results found. Run the engine first.")
            return
            
        with open(scores_path) as f:
            all_results = json.load(f)
        
        errors = []
        for task_name, result in all_results.items():
            predicted = result.get('predicted_category')
            expected = result.get('expected_category')
            if expected and predicted != expected:
                errors.append((task_name, predicted, expected))
        
        if not errors:
            print("🎉 No errors found!")
            return
            
        print(f"\n❌ ERROR ANALYSIS ({len(errors)} errors):")
        print("=" * 50)
        
        # Group errors by pattern
        error_patterns = {}
        for task_name, predicted, expected in errors:
            pattern = f"{expected} → {predicted}"
            if pattern not in error_patterns:
                error_patterns[pattern] = []
            error_patterns[pattern].append(task_name)
        
        # Show most common error patterns
        sorted_patterns = sorted(error_patterns.items(), key=lambda x: -len(x[1]))
        for pattern, tasks in sorted_patterns[:5]:  # Top 5 error patterns
            print(f"\n{pattern}: {len(tasks)} cases")
            for task in tasks[:3]:  # Show 3 examples
                print(f"  - {task}")
            if len(tasks) > 3:
                print(f"  ... and {len(tasks) - 3} more")
        
        return errors

    def get_performance_summary(self):
        """Get detailed performance summary"""
        scores_path = self.output_folder / "evaluated_scores.json"
        if not scores_path.exists():
            print("❌ No results found. Run the engine first.")
            return
            
        with open(scores_path) as f:
            all_results = json.load(f)
        
        # Calculate metrics
        total_with_expected = sum(1 for r in all_results.values() if r.get('expected_category'))
        correct = sum(1 for r in all_results.values() 
                     if r.get('expected_category') and r.get('predicted_category') == r.get('expected_category'))
        
        if total_with_expected == 0:
            print("No manual categorizations found for comparison.")
            return
            
        accuracy = correct / total_with_expected * 100
        
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print("=" * 30)
        print(f"Overall Accuracy: {accuracy:.1f}% ({correct}/{total_with_expected})")
        print(f"AI Calls Made: {self.ai_calls_made}")
        print(f"Total Hypotheses: {self.total_hypotheses_generated}")
        print(f"Avg Hypotheses/Task: {self.total_hypotheses_generated / len(all_results):.1f}")
        
        # Category-wise performance
        category_performance = {cat: {"correct": 0, "total": 0} for cat in self.categories}
        
        for result in all_results.values():
            expected = result.get('expected_category')
            predicted = result.get('predicted_category')
            
            if expected and expected in category_performance:
                category_performance[expected]["total"] += 1
                if predicted == expected:
                    category_performance[expected]["correct"] += 1
        
        print(f"\nCategory-wise Performance:")
        for cat, perf in category_performance.items():
            if perf["total"] > 0:
                cat_acc = perf["correct"] / perf["total"] * 100
                print(f"  {cat:12}: {cat_acc:5.1f}% ({perf['correct']}/{perf['total']})")
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total_with_expected,
            "ai_calls": self.ai_calls_made,
            "total_hypotheses": self.total_hypotheses_generated,
            "category_performance": category_performance
        }

    def view_interpretations(self, task_name=None):
        """View saved problem interpretations"""
        interpretations_path = self.output_folder / "problem_interpretations.json"
        
        if not interpretations_path.exists():
            print("❌ No interpretations found. Run the engine first.")
            return
        
        with open(interpretations_path) as f:
            interpretations = json.load(f)
        
        if not interpretations:
            print("No interpretations available.")
            return
        
        if task_name:
            # Show specific task interpretation
            if task_name in interpretations:
                data = interpretations[task_name]
                print(f"\n🔍 INTERPRETATION for {task_name}:")
                print("=" * 50)
                print(data['full_interpretation'])
                print(f"\nGenerated: {data['generated_at']}")
            else:
                print(f"❌ No interpretation found for task: {task_name}")
        else:
            # Show all interpretations summary
            print(f"\n📚 ALL PROBLEM INTERPRETATIONS ({len(interpretations)} tasks):")
            print("=" * 60)
            
            for i, (task, data) in enumerate(interpretations.items(), 1):
                metrics = data['confidence_metrics']
                confidence = "🟢 High" if metrics['best_pass_rate'] >= 0.8 else "🟡 Medium" if metrics['best_pass_rate'] >= 0.6 else "🔴 Low"
                
                print(f"\n{i}. {task}")
                print(f"   {confidence} confidence | Category: {metrics['best_category']} | Pass rate: {metrics['best_pass_rate']*100:.0f}%")
                print(f"   {data['plain_text_interpretation'][:100]}{'...' if len(data['plain_text_interpretation']) > 100 else ''}")
            
            print(f"\n💡 Use view_interpretations('task_name.json') to see full interpretation for a specific task.")

    def export_interpretations_summary(self):
        """Export a clean summary of interpretations for easy reference"""
        interpretations_path = self.output_folder / "problem_interpretations.json"
        
        if not interpretations_path.exists():
            print("❌ No interpretations found. Run the engine first.")
            return
        
        with open(interpretations_path) as f:
            interpretations = json.load(f)
        
        # Create a clean summary
        summary = {
            "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_problems": len(interpretations),
            "problems": {}
        }
        
        for task_name, data in interpretations.items():
            metrics = data['confidence_metrics']
            
            # Confidence level
            confidence_level = "High" if metrics['best_pass_rate'] >= 0.8 else "Medium" if metrics['best_pass_rate'] >= 0.6 else "Low"
            
            summary["problems"][task_name] = {
                "what_it_does": data['plain_text_interpretation'],
                "confidence": confidence_level,
                "category": metrics['best_category'],
                "success_rate": f"{metrics['best_pass_rate']*100:.0f}%",
                "num_hypotheses_analyzed": metrics['num_hypotheses']
            }
        
        # Save clean summary
        summary_path = self.output_folder / "interpretations_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Clean interpretations summary saved to: {summary_path.resolve()}")
        return summary_path
    
    def generate_interpretations_from_existing_data(self):
        """
        Generate problem interpretations using existing hypothesis data
        without rerunning the entire engine
        """
        # Check if hypotheses file exists
        hypotheses_path = self.output_folder / "generated_hypotheses.json"
        
        if not hypotheses_path.exists():
            print("❌ No existing hypothesis data found. You need to run the engine first.")
            return False
        
        print("📚 Loading existing hypothesis data...")
        with open(hypotheses_path) as f:
            all_hypotheses = json.load(f)
        
        print(f"✅ Found hypothesis data for {len(all_hypotheses)} tasks")
        
        # Generate interpretations for each task
        print("🤖 Generating AI interpretations from existing hypotheses...")
        
        interpretations_generated = 0
        
        for task_name, hypotheses in all_hypotheses.items():
            if not hypotheses:  # Skip tasks with no hypotheses
                continue
                
            try:
                print(f"   🔄 Generating interpretation for {task_name}...")
                
                # Generate interpretation using ChatGPT
                interpretation = self.generate_problem_interpretation(hypotheses, task_name)
                
                # Store interpretation
                self.problem_interpretations[task_name] = {
                    'interpretation': interpretation,
                    'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'num_hypotheses': len(hypotheses),
                    'best_pass_rate': max([h.get('pass_rate', 0) for h in hypotheses]) if hypotheses else 0,
                    'best_category': max(hypotheses, key=lambda x: x.get('pass_rate', 0)).get('category', 'Unknown') if hypotheses else 'Unknown'
                }
                
                interpretations_generated += 1
                
                # Small delay to respect rate limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"   ❌ Error generating interpretation for {task_name}: {str(e)}")
                continue
        
        # Save all interpretations
        if interpretations_generated > 0:
            interpretations_file = self.save_problem_interpretations()
            summary_file = self.export_interpretations_summary()
            
            print(f"✅ Generated {interpretations_generated} interpretations!")
            print(f"📁 Saved to: {interpretations_file}")
            print(f"📁 Summary: {summary_file}")
            
            return True
        else:
            print("❌ No interpretations could be generated.")
            return False

    # Also add this convenience function to the main script
    def generate_interpretations_only():
        """Standalone function to generate interpretations from existing data"""
        import os
        # Set your OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY", None)
        
        if not openai_api_key:
            print("⚠️ Warning: No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
            return False
        
        # Paths  
        arc_folder = "./MINI-ARC/data/MiniARC"
        data_folder = r"./generated data/Test #3 17.08"
        
        # Initialize engine (lightweight - just for interpretation generation)
        engine = HybridRuleEngine(
            arc_folder, 
            data_folder, 
            openai_api_key,
            max_concurrent_requests=2,
            rpm_limit=150
        )
        
        # Generate interpretations from existing hypothesis data
        success = engine.generate_interpretations_from_existing_data()
        
        if success:
            print("\n🧠 Viewing generated interpretations:")
            engine.view_interpretations()
            
            print("\n💡 You can now use:")
            print("  - engine.view_interpretations() to see all interpretations")
            print("  - engine.view_interpretations('task_name.json') to see specific task")
            print(f"  - Check {data_folder}/interpretations_summary.json for clean summaries")
        
        return success