from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dsl import objects
from solomonoff import calculate_solomonoff_score
from rules import ALL_RULES
from categories import CATEGORIES
from utils.complexity import rule_complexity  # counts tokens of NL description
from utils.visualisation import compare_multiple_pairs, display_rule_results, plot_solomonoff_scores
from openai import OpenAI  # NEW: import OpenAI client
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        self.manual_results = self._load_manual_results()
        self.client = OpenAI(api_key=openai_api_key) if openai_api_key else OpenAI()

    def _load_manual_results(self):
        path = Path("manual_categorization.json")
        return json.load(path.open()) if path.exists() else {}

    def query_gpt_batch(self, rule_descriptions, input_grid, output_grid, retries=2, timeout=5):
        """
        Batch GPT call: asks GPT to evaluate all hypotheses at once,
        returns list of booleans corresponding to each rule.
        """
        prompt = f"""
        You are an expert at solving Abstraction and Reasoning Corpus (ARC) tasks.

        Given the following example:

        Input grid:
        {input_grid}

        Output grid:
        {output_grid}

        Evaluate the following hypotheses about the transformation from input to output:

        """ + "\n".join(f"{i+1}. {desc}" for i, desc in enumerate(rule_descriptions)) + """

        For each hypothesis, answer ONLY 'True' or 'False' on a separate line, in order.
        """

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10 * len(rule_descriptions),
                    n=1,
                    stop=None,
                    timeout=timeout  # if supported by your client
                )
                answers = response.choices[0].message.content.strip().splitlines()
                return [ans.strip().lower().startswith("true") for ans in answers]
            except Exception as e:
                print(f"GPT batch query error on attempt {attempt+1}: {e}")
                if attempt == retries - 1:
                    return [False] * len(rule_descriptions)
                time.sleep(1)

    def evaluate_category(self, task, category, rules):
        if "train" not in task or not task["train"]:
            print(f"❌ Missing or empty 'train' in task: {task}")
            return 0.0, {}

        first_pair = task["train"][0]
        inp_grid = np.array(first_pair["input"])
        out_grid = np.array(first_pair["output"])

        if not rules:
            return 0.0, {}

        rule_descriptions = [rule[0] for rule in rules]
        priors = [rule[1] for rule in rules]

        results = self.query_gpt_batch(rule_descriptions, inp_grid.tolist(), out_grid.tolist())

        total_score = 0.0
        passed_rules = {}

        for passed, prior, rule_description in zip(results, priors, rule_descriptions):
            passed_rules[rule_description] = passed
            if passed:
                complexity = rule_complexity(rule_description)
                total_score += calculate_solomonoff_score([True], prior, complexity)

        return total_score, passed_rules


    def run(self, save_results=True):
        self.output_folder.mkdir(parents=True, exist_ok=True)
        manual_results = self.manual_results
        scores_path = self.output_folder / "evaluated_scores.json"
        if scores_path.exists():
            scores_path.unlink()
            print(f"🗑️ Deleted existing {scores_path.name}")

        tasks = list(self.data_folder.glob("*.json"))
        total_tasks = len(tasks)
        all_results = {}
        correct_count = 0
        correct_tasks = []

        def process_task(task_path_idx):
            idx, task_path = task_path_idx
            print(f"Start task {idx}: {task_path.name}")
            start = time.time()

            with open(task_path) as f:
                task = json.load(f)

            task_name = task_path.name
            self.task_data[task_name] = task

            category_scores = {}
            detailed_results = {}

            for category in CATEGORIES:
                score, passed_rules = self.evaluate_category(task, category, ALL_RULES.get(category, []))
                category_scores[category] = score
                detailed_results[category] = passed_rules

            scores = np.array(list(category_scores.values()))
            normalized_scores = softmax(scores)

            normalized_category_scores = {
                category: normalized_scores[idx] for idx, category in enumerate(CATEGORIES)
            }

            best_category = max(normalized_category_scores, key=normalized_category_scores.get)
            task['predicted_scores'] = normalized_category_scores
            task['predicted_categories'] = [best_category] * len(task.get("train", []))
            task['detailed_rule_results'] = detailed_results   # <--- save here

            if self.manual_results and task_name in self.manual_results:
                task['expected_category'] = self.manual_results[task_name]

            if save_results:
                output_path = self.output_folder / f"{task_path.stem}_evaluated.json"
                with open(output_path, "w") as out_f:
                    json.dump(task, out_f, indent=2)

            expected_category = self.manual_results.get(task_name) if self.manual_results else None
            duration = time.time() - start
            print(f"Finished task {idx}: {task_name} in {duration:.1f}s")

            return (idx, task_name, best_category, expected_category, normalized_category_scores)


        from concurrent.futures import as_completed

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_task, item): item for item in enumerate(tasks)}
            with tqdm(total=len(tasks), desc="Processing tasks") as pbar:
                for future in as_completed(futures):
                    idx, task_name, best_cat, expected_cat, norm_scores = future.result()
                    all_results[task_name] = {
                        "predicted_category": best_cat,
                        "expected_category": expected_cat,
                        "scores": norm_scores
                    }
                    if manual_results and expected_cat == best_cat:
                        correct_count += 1
                        correct_tasks.append((idx, task_name))
                    pbar.update(1)

        print("Evaluation complete. Results saved.")

        if save_results:
            with open(scores_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\n✅ Scores saved to: {scores_path.resolve()}")

        if manual_results:
            accuracy = correct_count / len(manual_results) * 100
            print(f"\n🤖 AI vs 👤 Human categorization accuracy: {accuracy:.2f}% ({correct_count}/{len(manual_results)})")
            print("\nCorrectly matched tasks:")
            for idx, task in sorted(correct_tasks):
                print(f" - [#{idx}] {task}")


    def manual_categorise(self):
            import matplotlib.pyplot as plt

            self.output_folder.mkdir(parents=True, exist_ok=True)
            manual_path = Path("manual_categorization.json")

            tasks = list(self.data_folder.glob("*.json"))
            manual_results = {}

            print("\nManual Categorization Mode")
            print("Categories:")
            for i, cat in enumerate(CATEGORIES):
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
                        elif inp.isdigit() and 0 <= int(inp) < len(CATEGORIES):
                            manual_results[task_name] = CATEGORIES[int(inp)]
                            print(f"✔️ Saved: {task_name} → {CATEGORIES[int(inp)]}")
                            break
                        else:
                            print(f"Invalid input. Please enter a number between 0 and {len(CATEGORIES)-1}, or 's' to skip.")
                    except KeyboardInterrupt:
                        print("\nExiting manual categorization.")
                        plt.close("all")
                        return

                plt.close("all")  # Close after each entry

            with open(manual_path, "w") as f:
                json.dump(manual_results, f, indent=2)

            print(f"\n✅ Manual categorization saved to: {manual_path}")

    def View(self, task_name=None):
        if not self.task_data:
            print("No tasks loaded. Run the engine first.")
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

        for task_name in task_names:
            print(f"\n--- Visualising Task: {task_name} ---")

            evaluated_path = self.output_folder / task_name.replace(".json", "_evaluated.json")
            if evaluated_path.exists():
                with open(evaluated_path) as f:
                    task = json.load(f)
            else:
                task = self.task_data.get(task_name)

            if task is None:
                print(f"Task '{task_name}' not found.")
                continue

            try:
                scores_path = self.output_folder / "evaluated_scores.json"
                with open(scores_path, "r") as f:
                    self.category_scores = json.load(f)
            except FileNotFoundError:
                print("Evaluated scores file not found. Please run the engine first.")
                return

            pairs = [(np.array(pair["input"]), np.array(pair["output"])) for pair in task["train"]]
            predicted_categories = task.get("predicted_categories", [])
            expected_category = task.get("expected_category", None)

            compare_multiple_pairs(
                pairs,
                task_id=task_name,
                predicted_categories=predicted_categories,
                expected_category=expected_category
            )

            # Print predicted vs expected category
            pred_cat = predicted_categories[0] if predicted_categories else "N/A"
            print(f"\nPredicted category: {pred_cat}")
            print(f"Expected category: {expected_category if expected_category else 'N/A'}")

            # Print all passing rules across all categories
            detailed_rule_results = task.get("detailed_rule_results", {})
            if detailed_rule_results:
                print("\nAll Passing Rules Across All Categories:")
                found_any = False
                for category, rules_results in detailed_rule_results.items():
                    passing_rules = [desc for desc, passed in rules_results.items() if passed]
                    if passing_rules:
                        found_any = True
                        print(f"\nCategory: {category}")
                        for rule_desc in passing_rules:
                            print(f" - ✅ {rule_desc}")
                if not found_any:
                    print("No passing rules found across any category.")
            else:
                print("No detailed rule results found.")

            score_dict = self.category_scores.get(task_name, {})
            if not score_dict:
                print("No score data found for this task.")
                continue

            plot_solomonoff_scores(score_dict)
            self.plot_task_objects(task_name)
            plt.show()



    def plot_task_objects(self, task_name):
        if task_name not in self.task_data:
            print(f"Task '{task_name}' not found in loaded data.")
            return

        task = self.task_data[task_name]
        train_pairs = task.get("train", [])

        for idx, pair in enumerate(train_pairs):
            for mode in ["input", "output"]:
                grid = tuple(tuple(row) for row in pair[mode])
                objs = objects(grid=grid, univalued=True, diagonal=True, without_bg=True)
                fig, ax = plt.subplots()
                ax.imshow(grid, cmap="tab20", interpolation="none")

                for obj in objs:
                    indices = [loc for _, loc in obj]
                    rows, cols = zip(*indices)
                    min_row, max_row = min(rows), max(rows)
                    min_col, max_col = min(cols), max(cols)
                    rect = patches.Rectangle(
                        (min_col - 0.5, min_row - 0.5),
                        max_col - min_col + 1,
                        max_row - min_row + 1,
                        linewidth=2,
                        edgecolor='red',
                        facecolor='none'
                    )
                    ax.add_patch(rect)

                ax.set_title(f"Task: {task_name} | Pair #{idx} | {mode.capitalize()}")
                plt.axis("off")
