from tqdm import tqdm
import numpy as np
from pathlib import Path
from solomonoff import calculate_solomonoff_score
from utils.visualisation import *
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dsl import *
from ai_hypotheses_generator import AIHypothesisGenerator
import traceback
from typing import List, Dict, Any, Tuple
import time

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class RuleEngine:
    def __init__(self, data_folder, output_folder, openai_api_key=None):
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
        
        # Performance tracking
        self.ai_calls_made = 0
        self.cache_hits = 0

    def load_tasks(self):
        """Load all tasks from the data folder"""
        tasks = list(self.data_folder.glob("*.json"))
        for task_path in tasks:
            with open(task_path) as f:
                task = json.load(f)
            task_name = task_path.name
            self.task_data[task_name] = task
        print(f"Loaded {len(self.task_data)} tasks")

    def manual_categorize(self):
        """Manual categorization interface"""
        import matplotlib.pyplot as plt

        self.output_folder.mkdir(parents=True, exist_ok=True)
        manual_path = Path("manual_categorization.json")

        tasks = list(self.data_folder.glob("*.json"))
        manual_results = {}

        print("\nManual Categorization Mode")
        print("Categories:")
        for i, cat in enumerate(self.categories):
            print(f"{i}: {cat}")

        plt.ion()  # Turn on interactive mode

        for idx, task_path in enumerate(tasks):
            task_name = task_path.name
            with open(task_path) as f:
                task = json.load(f)

            pairs = [(np.array(pair["input"]), np.array(pair["output"])) for pair in task["train"]]

            # Visualize the task
            fig = compare_multiple_pairs(pairs, task_id=task_name)
            plt.pause(0.001)  # Show non-blocking plot

            # Prompt for input
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

            plt.close("all")  # Close after each entry

        with open(manual_path, "w") as f:
            json.dump(manual_results, f, indent=2)

        print(f"\n✅ Manual categorization saved to: {manual_path}")

    def generate_and_test_hypotheses_fast(self, task: Dict[str, Any], use_ai_selectively: bool = True) -> List[Dict[str, Any]]:
        """Fast hypothesis generation and testing"""
        if "train" not in task or not task["train"]:
            return []

        # Use first training pair for hypothesis generation
        first_pair = task["train"][0]
        input_grid = np.array(first_pair["input"])
        output_grid = np.array(first_pair["output"])

        # Decide if we need expensive AI based on pattern complexity
        pattern_type = self.hypothesis_generator.quick_pattern_analysis(input_grid, output_grid)
        use_ai = use_ai_selectively and pattern_type == "complex"

        # Generate hypotheses (mostly templates, AI only when needed)
        hypotheses = self.hypothesis_generator.generate_hypotheses_optimized(
            input_grid, output_grid, use_ai=use_ai
        )
        
        if use_ai:
            self.ai_calls_made += 1
        else:
            self.cache_hits += 1

        tested_hypotheses = []

        for hypothesis in hypotheses:
            try:
                # Test the hypothesis on all training pairs (fast execution)
                test_results = []
                for pair in task["train"]:
                    inp = np.array(pair["input"])
                    out = np.array(pair["output"])
                    
                    # Get objects if available (cached)
                    try:
                        inp_objs = objects(tuple(tuple(row) for row in pair["input"]), True, True, True)
                        out_objs = objects(tuple(tuple(row) for row in pair["output"]), True, True, True)
                    except:
                        inp_objs = None
                        out_objs = None
                    
                    # Execute the test code (fast)
                    result = self.execute_hypothesis_test_fast(hypothesis['test_code'], inp, out, inp_objs, out_objs)
                    test_results.append(result)

                # Calculate complexity and score
                complexity = self.hypothesis_generator.rule_complexity(hypothesis['description'])
                prior = hypothesis.get('prior_probability', 0.1)
                
                # Calculate Solomonoff score
                score = calculate_solomonoff_score(test_results, prior, complexity)
                
                hypothesis_result = {
                    **hypothesis,
                    'test_results': test_results,
                    'complexity': complexity,
                    'solomonoff_score': score,
                    'passed_all': all(test_results),
                    'pass_rate': sum(test_results) / len(test_results) if test_results else 0
                }
                
                tested_hypotheses.append(hypothesis_result)
                
            except Exception as e:
                # Skip failed hypotheses silently for speed
                continue

        return tested_hypotheses

    def execute_hypothesis_test_fast(self, test_code: str, input_grid: np.ndarray, 
                                   output_grid: np.ndarray, input_objects=None, 
                                   output_objects=None) -> bool:
        """Fast hypothesis test execution with minimal overhead"""
        try:
            # Minimal namespace for speed
            namespace = {
                'np': np,
                'input_grid': input_grid,
                'output_grid': output_grid,
                'input_objects': input_objects,
                'output_objects': output_objects
            }
            
            # Execute the test code
            exec(test_code, namespace)
            
            # Call the test function
            result = namespace['test_hypothesis'](input_grid, output_grid, input_objects, output_objects)
            return bool(result)
            
        except:
            # Fail fast
            return False

    def evaluate_task_with_ai_fast(self, task: Dict[str, Any]) -> Dict[str, float]:
        """Fast task evaluation with selective AI usage"""
        hypotheses = self.generate_and_test_hypotheses_fast(task, use_ai_selectively=True)
        
        # Group hypotheses by category and calculate scores
        category_scores = {category: 0.0 for category in self.categories}
        
        for hypothesis in hypotheses:
            category = hypothesis['category']
            if category in category_scores:
                # Weight by pass rate and confidence, bonus for passing all tests
                base_weight = hypothesis['pass_rate'] * hypothesis.get('confidence', 0.5)
                if hypothesis['passed_all']:
                    base_weight *= 2.0  # Bonus for perfect hypotheses
                    
                # Use negative score (lower is better for Solomonoff) 
                contribution = -hypothesis['solomonoff_score'] * base_weight
                category_scores[category] += contribution

        return category_scores

    def run(self, save_results=True, use_ai_hypotheses=True, max_ai_tasks=None):
        """Optimized run with cost control"""
        start_time = time.time()
        self.output_folder.mkdir(parents=True, exist_ok=True)

        manual_path = Path("manual_categorization.json")
        manual_results = {}
        if manual_path.exists():
            with open(manual_path) as f:
                manual_results = json.load(f)

        scores_path = self.output_folder / "evaluated_scores.json"
        hypotheses_path = self.output_folder / "generated_hypotheses.json"
        
        # Clean up old files
        if scores_path.exists():
            scores_path.unlink()

        tasks = list(self.data_folder.glob("*.json"))
        total_tasks = len(tasks)
        
        # Limit AI usage if specified
        if max_ai_tasks and max_ai_tasks < total_tasks:
            print(f"💰 Cost control: Using AI for max {max_ai_tasks} tasks, templates for others")
        
        all_results = {}
        all_hypotheses = {}
        correct_count = 0
        correct_tasks = []

        ai_tasks_used = 0
        
        for idx, task_path in tqdm(enumerate(tasks), total=total_tasks, desc="Processing tasks (optimized)", unit="task"):
            with open(task_path) as f:
                task = json.load(f)

            task_name = task_path.name
            self.task_data[task_name] = task

            # Control AI usage
            use_ai_for_task = (use_ai_hypotheses and 
                             (max_ai_tasks is None or ai_tasks_used < max_ai_tasks))

            if use_ai_for_task:
                # Use AI-powered analysis
                category_scores = self.evaluate_task_with_ai_fast(task)
                hypotheses = self.generate_and_test_hypotheses_fast(task, use_ai_selectively=True)
                all_hypotheses[task_name] = hypotheses
                ai_tasks_used += 1
            else:
                # Use template-only analysis (very fast)
                category_scores = self.evaluate_task_with_ai_fast(task)
                # Generate templates only
                first_pair = task["train"][0]
                input_grid = np.array(first_pair["input"])
                output_grid = np.array(first_pair["output"])
                hypotheses = self.hypothesis_generator.generate_hypotheses_optimized(
                    input_grid, output_grid, use_ai=False
                )
                # Test the template hypotheses
                tested_hypotheses = []
                for hypothesis in hypotheses:
                    try:
                        test_results = []
                        for pair in task["train"]:
                            inp = np.array(pair["input"])
                            out = np.array(pair["output"])
                            try:
                                inp_objs = objects(tuple(tuple(row) for row in pair["input"]), True, True, True)
                                out_objs = objects(tuple(tuple(row) for row in pair["output"]), True, True, True)
                            except:
                                inp_objs = None
                                out_objs = None
                            result = self.execute_hypothesis_test_fast(hypothesis['test_code'], inp, out, inp_objs, out_objs)
                            test_results.append(result)
                        
                        complexity = self.hypothesis_generator.rule_complexity(hypothesis['description'])
                        prior = hypothesis.get('prior_probability', 0.1)
                        score = calculate_solomonoff_score(test_results, prior, complexity)
                        
                        hypothesis_result = {
                            **hypothesis,
                            'test_results': test_results,
                            'complexity': complexity,
                            'solomonoff_score': score,
                            'passed_all': all(test_results),
                            'pass_rate': sum(test_results) / len(test_results) if test_results else 0
                        }
                        tested_hypotheses.append(hypothesis_result)
                    except:
                        continue
                        
                all_hypotheses[task_name] = tested_hypotheses

            # Normalize scores
            scores = np.array(list(category_scores.values()))
            if np.sum(np.abs(scores)) > 0:
                normalized_scores = softmax(scores)
            else:
                normalized_scores = np.ones(len(scores)) / len(scores)

            normalized_category_scores = {
                category: normalized_scores[idx] for idx, category in enumerate(self.categories)
            }

            best_category = max(normalized_category_scores, key=normalized_category_scores.get)
            task['predicted_scores'] = normalized_category_scores
            task['predicted_categories'] = [best_category] * len(task["train"])
            
            if manual_results and task_name in manual_results:
                task['expected_category'] = manual_results[task_name]

            # Add hypotheses to task data for saving
            if task_name in all_hypotheses:
                task['generated_hypotheses'] = all_hypotheses[task_name]
            
            if save_results:
                output_path = self.output_folder / f"{task_path.stem}_evaluated.json"
                with open(output_path, "w") as out_f:
                    json.dump(task, out_f, indent=2, default=str)

            expected_category = manual_results.get(task_name) if manual_results else None
            all_results[task_name] = {
                "predicted_category": best_category,
                "expected_category": expected_category,
                "scores": normalized_category_scores
            }

            # Compare with manual if available
            if manual_results and task_name in manual_results:
                manual_cat = manual_results[task_name]
                if manual_cat == best_category:
                    correct_count += 1
                    correct_tasks.append((idx, task_name))

        end_time = time.time()
        print(f"⚡ Completed in {end_time - start_time:.1f} seconds")
        print(f"💰 AI calls made: {self.ai_calls_made}, Template usage: {self.cache_hits}")

        if save_results:
            with open(scores_path, "w") as f:
                json.dump(all_results, f, indent=2)
            
            with open(hypotheses_path, "w") as f:
                json.dump(all_hypotheses, f, indent=2, default=str)
            
            print(f"\n✅ Results saved to: {scores_path.resolve()}")

        if manual_results:
            accuracy = correct_count / len(manual_results) * 100
            print(f"\n🎯 Accuracy: {accuracy:.2f}% ({correct_count}/{len(manual_results)})")
            
            if correct_tasks:
                print("✅ Sample correct predictions:")
                for idx, task in correct_tasks[:5]:  # Show first 5
                    print(f" - [#{idx}] {task}")

    def View(self, task_name=None):
        """Enhanced visualization with performance info"""
        if not self.task_data:
            print("No tasks loaded. Run load_tasks() first.")
            return

        task_names = list(self.task_data.keys())

        # Handle integer index input
        if isinstance(task_name, int):
            if 0 <= task_name < len(task_names):
                task_names = [task_names[task_name]]
            else:
                print(f"Invalid index: {task_name}. Must be between 0 and {len(task_names)-1}.")
                return
        elif isinstance(task_name, str):
            task_names = [task_name]
        elif task_name is None:
            task_names = task_names[:1]  # Show first task by default

        for task_name in task_names:
            print(f"\n🔍 Task: {task_name}")

            evaluated_path = self.output_folder / task_name.replace(".json", "_evaluated.json")
            if evaluated_path.exists():
                with open(evaluated_path) as f:
                    task = json.load(f)
            else:
                task = self.task_data.get(task_name)

            if task is None:
                print(f"Task '{task_name}' not found.")
                continue

            # Load hypotheses if available
            hypotheses_path = self.output_folder / "generated_hypotheses.json"
            if hypotheses_path.exists():
                with open(hypotheses_path) as f:
                    all_hypotheses = json.load(f)
                    task_hypotheses = all_hypotheses.get(task_name, [])
                    
                print(f"\n🧠 Top Hypotheses ({len(task_hypotheses)}):")
                # Sort by score and show best ones
                sorted_hyp = sorted(task_hypotheses, key=lambda x: -x.get('pass_rate', 0))
                for i, hyp in enumerate(sorted_hyp[:3]):  # Show top 3
                    source = "🤖" if hyp.get('source') == 'ai' else "📋"
                    print(f"{i+1}. {source} [{hyp['category']}] {hyp['description']}")
                    print(f"   ✅ Pass: {hyp.get('pass_rate', 0):.2f} | Score: {hyp.get('solomonoff_score', 0):.3f}")

            # Visualize the task
            pairs = [(np.array(pair["input"]), np.array(pair["output"])) for pair in task["train"]]
            predicted_categories = task.get("predicted_categories", [])
            expected_category = task.get("expected_category", None)
            
            try:
                compare_multiple_pairs(
                    pairs, 
                    task_id=task_name, 
                    predicted_categories=predicted_categories,
                    expected_category=expected_category
                )

                # Show scores
                scores_path = self.output_folder / "evaluated_scores.json"
                if scores_path.exists():
                    with open(scores_path, "r") as f:
                        self.category_scores = json.load(f)
                        
                    score_dict = self.category_scores.get(task_name, {})
                    if score_dict:
                        plot_solomonoff_scores(score_dict)

                plt.show()
                
            except Exception as e:
                print(f"Visualization error: {e}")
                continue