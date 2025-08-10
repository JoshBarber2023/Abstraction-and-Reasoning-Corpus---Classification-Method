# rule_engine.py
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
import threading
from collections import deque
import random
import re


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class RuleEngine:
    def __init__(
        self,
        data_folder,
        output_folder,
        openai_api_key=None,
        max_concurrent_requests: int = 4,  # reduce concurrency to avoid bursting
        rpm_limit: int = 400  # requests per minute limit (configure to your org quota)
    ):
        self.data_folder = Path(data_folder)
        self.output_folder = Path(output_folder)
        self.task_data = {}
        self.category_scores = {}
        self.manual_results = self._load_manual_results()
        self.client = OpenAI(api_key=openai_api_key) if openai_api_key else OpenAI()

        # Rate limiting and concurrency controls
        self.max_concurrent_requests = max_concurrent_requests
        self.rpm_limit = rpm_limit
        self._request_lock = threading.Lock()
        self._recent_call_timestamps = deque()  # stores monotonic() timestamps of recent calls
        self._concurrent_semaphore = threading.BoundedSemaphore(value=self.max_concurrent_requests)

    def _load_manual_results(self):
        path = Path("manual_categorization.json")
        return json.load(path.open()) if path.exists() else {}

    def _wait_for_rate_slot(self):
        """
        Simple sliding-window RPM rate limiter. Blocks until it's safe to make a request.
        """
        with self._request_lock:
            now = time.monotonic()
            # remove timestamps older than 60 seconds
            while self._recent_call_timestamps and now - self._recent_call_timestamps[0] > 60.0:
                self._recent_call_timestamps.popleft()

            if len(self._recent_call_timestamps) < self.rpm_limit:
                # we can proceed
                self._recent_call_timestamps.append(now)
                return
            else:
                # compute wait time until oldest timestamp falls outside 60s window
                earliest = self._recent_call_timestamps[0]
                sleep_for = 60.0 - (now - earliest) + 0.01
        # release lock before sleeping
        time.sleep(sleep_for)
        # recursive: try again (rarely more than once)
        return self._wait_for_rate_slot()

    def query_gpt_batch(self, rule_descriptions, input_grid, output_grid, retries=5, timeout=10):
        """
        Batch GPT call with:
        - grid normalization for dsl.objects()
        - concurrency/semaphore control to limit simultaneous API calls
        - per-minute RPM sliding-window limiter
        - exponential backoff with jitter on rate-limit or transient errors

        Returns: list[bool] corresponding to each rule description.
        """
        def _normalize_for_dsl(grid):
            arr = np.array(grid, dtype=int)
            return tuple(tuple(int(v) for v in row) for row in arr.tolist())

        # Normalize grids for DSL
        try:
            input_grid_dsl = _normalize_for_dsl(input_grid)
            output_grid_dsl = _normalize_for_dsl(output_grid)
        except Exception as e:
            print(f"[Warning] grid normalization failed: {e}")
            try:
                input_grid_dsl = tuple(tuple(int(v) for v in row) for row in input_grid)
                output_grid_dsl = tuple(tuple(int(v) for v in row) for row in output_grid)
            except Exception:
                input_grid_dsl = tuple(tuple(row) for row in input_grid)
                output_grid_dsl = tuple(tuple(row) for row in output_grid)

        # Get objects for DSL
        try:
            input_objs = objects(input_grid_dsl, True, True, True)
        except Exception as e:
            print(f"[Warning] objects() failed on input grid: {e}")
            input_objs = frozenset()

        try:
            output_objs = objects(output_grid_dsl, True, True, True)
        except Exception as e:
            print(f"[Warning] objects() failed on output grid: {e}")
            output_objs = frozenset()

        def format_objects(obj_set):
            if not obj_set:
                return "None"
            lines = []
            for idx, obj in enumerate(obj_set, start=1):
                colors = sorted({v for v, _ in obj})
                coords = sorted([loc for _, loc in obj])
                lines.append(f"Obj {idx}: colors={colors}, size={len(coords)}, coords={coords}")
            return "\n".join(lines)

        input_obj_str = format_objects(input_objs)
        output_obj_str = format_objects(output_objs)

        prompt = (
            "You are an expert at solving Abstraction and Reasoning Corpus (ARC) tasks.\n\n"
            "Here is the example you will analyze:\n\n"
            f"Input grid:\n{input_grid}\n\n"
            f"Input objects (excluding background):\n{input_obj_str}\n\n"
            f"Output grid:\n{output_grid}\n\n"
            f"Output objects (excluding background):\n{output_obj_str}\n\n"
            "Evaluate the following hypotheses about the transformation from input to output:\n\n"
            + "\n".join(f"{i+1}. {desc}" for i, desc in enumerate(rule_descriptions))
            + "\n\nFor each hypothesis, answer ONLY 'True' or 'False' on a separate line, in order.\n"
        )

        acquired = self._concurrent_semaphore.acquire(timeout=30)
        if not acquired:
            print("[Warning] Could not acquire concurrent semaphore; proceeding without it.")
        try:
            attempt = 0
            while attempt < retries:
                attempt += 1
                self._wait_for_rate_slot()
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=10 * len(rule_descriptions),
                        n=1,
                        stop=None,
                        timeout=timeout
                    )
                    answers = response.choices[0].message.content.strip().splitlines()
                    return [ans.strip().lower().startswith("true") for ans in answers]
                except Exception as e:
                    err_text = str(e).lower()
                    is_rate = (
                        "rate" in err_text and (
                            "limit" in err_text or
                            "rate_limit" in err_text or
                            "rate_limit_exceeded" in err_text or
                            "requests per min" in err_text
                        )
                    )
                    is_transient = isinstance(e, TimeoutError) or "timeout" in err_text or "temporar" in err_text or "503" in err_text

                    if is_rate or is_transient:
                        backoff_base = 0.5 * (2 ** (attempt - 1))
                        jitter = random.uniform(0, 0.5)
                        sleep_time = min(backoff_base + jitter, 60.0)
                        print(f"GPT batch query error on attempt {attempt}: {e} -- retrying after {sleep_time:.2f}s")
                        time.sleep(sleep_time)
                        continue
                    else:
                        print(f"Non-retriable GPT error: {e}")
                        return [False] * len(rule_descriptions)
            print("Exceeded retry attempts for GPT call; returning all-False.")
            return [False] * len(rule_descriptions)
        finally:
            try:
                if acquired:
                    self._concurrent_semaphore.release()
            except Exception:
                pass

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

    ####################################################################################
    # NEW METHOD: Generate GPT hypotheses independently for top categories and evaluate
    ####################################################################################
    def generate_and_evaluate_hypotheses_for_categories(self, task, categories, top_n=3, max_hypotheses=5, retries=3):
        """
        Given a task and a list of categories, ask GPT to independently generate
        hypotheses explaining why the task might belong to each category.

        Then, evaluate those hypotheses with GPT (True/False), compute
        Solomonoff scores and rank categories by summed scores.

        Returns:
            refined_scores: dict(category -> float score)
            detailed_hypotheses: dict(category -> list of (hypothesis:str, passed:bool, complexity:int, solomonoff_score:float))
        """
        if "train" not in task or not task["train"]:
            print("❌ Missing or empty 'train' in task for hypothesis generation.")
            return {}, {}

        first_pair = task["train"][0]
        inp_grid = np.array(first_pair["input"])
        out_grid = np.array(first_pair["output"])

        def _normalize_for_dsl(grid):
            arr = np.array(grid, dtype=int)
            return tuple(tuple(int(v) for v in row) for row in arr.tolist())

        # Normalize grids for DSL objects extraction
        try:
            input_grid_dsl = _normalize_for_dsl(inp_grid)
            output_grid_dsl = _normalize_for_dsl(out_grid)
        except Exception as e:
            print(f"[Warning] grid normalization failed during hypothesis generation: {e}")
            input_grid_dsl = tuple(tuple(int(v) for v in row) for row in inp_grid)
            output_grid_dsl = tuple(tuple(int(v) for v in row) for row in out_grid)

        # Extract objects for input and output grids
        try:
            input_objs = objects(input_grid_dsl, True, True, True)
        except Exception as e:
            print(f"[Warning] objects() failed on input grid during hypothesis generation: {e}")
            input_objs = frozenset()

        try:
            output_objs = objects(output_grid_dsl, True, True, True)
        except Exception as e:
            print(f"[Warning] objects() failed on output grid during hypothesis generation: {e}")
            output_objs = frozenset()

        def format_objects(obj_set):
            if not obj_set:
                return "None"
            lines = []
            for idx, obj in enumerate(obj_set, start=1):
                colors = sorted({v for v, _ in obj})
                coords = sorted([loc for _, loc in obj])
                lines.append(f"Obj {idx}: colors={colors}, size={len(coords)}, coords={coords}")
            return "\n".join(lines)

        input_obj_str = format_objects(input_objs)
        output_obj_str = format_objects(output_objs)

        # Limit categories to top_n
        top_categories = categories[:top_n]

        refined_scores = {}
        detailed_hypotheses = {}

        for category in top_categories:
            # Get category description text from ALL_RULES if possible
            rules_for_cat = ALL_RULES.get(category, [])
            if rules_for_cat:
                category_description = "\n".join([rule[0] for rule in rules_for_cat])
            else:
                category_description = "No detailed description available."

            # Build an explicit *category purpose* or *definition* block for GPT:
            category_definition_text = (
                f"This category ({category}) concerns the following key concepts:\n"
                f"{category_description}\n\n"
                "When generating hypotheses, you MUST ONLY describe transformations or properties\n"
                "that fit within this category's scope. Do NOT include hypotheses about other categories.\n"
                "Each hypothesis MUST explicitly reference the input and output objects listed below.\n"
                "Avoid vague statements; be precise and base hypotheses on the objects' colors, shapes, sizes, positions, or counts as relevant to this category.\n"
            )

            prompt = (
                "You are an expert solver of Abstraction and Reasoning Corpus (ARC) tasks.\n\n"
                + category_definition_text
                + "\n"
                "Here is the example you will analyze:\n\n"
                f"Input grid:\n{inp_grid.tolist()}\n\n"
                f"Input objects (excluding background):\n{input_obj_str}\n\n"
                f"Output grid:\n{out_grid.tolist()}\n\n"
                f"Output objects (excluding background):\n{output_obj_str}\n\n"
                f"Generate up to {max_hypotheses} numbered, detailed hypotheses about the transformation from input to output.\n"
                "Each hypothesis must be relevant to the category definition above.\n"
                "List hypotheses one per line, numbered.\n"
                "Do NOT judge their truth yet — only list them.\n"
            )

            acquired = self._concurrent_semaphore.acquire(timeout=30)
            if not acquired:
                print("[Warning] Could not acquire concurrent semaphore; proceeding without it.")
            hypotheses_text = ""
            try:
                attempt = 0
                while attempt < retries:
                    attempt += 1
                    self._wait_for_rate_slot()
                    try:
                        response = self.client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7,
                            max_tokens=350,
                            n=1,
                            stop=None,
                            timeout=15
                        )
                        hypotheses_text = response.choices[0].message.content.strip()
                        break
                    except Exception as e:
                        err_text = str(e).lower()
                        is_rate = (
                            "rate" in err_text and (
                                "limit" in err_text or
                                "rate_limit" in err_text or
                                "rate_limit_exceeded" in err_text or
                                "requests per min" in err_text
                            )
                        )
                        is_transient = isinstance(e, TimeoutError) or "timeout" in err_text or "temporar" in err_text or "503" in err_text

                        if is_rate or is_transient:
                            backoff_base = 0.5 * (2 ** (attempt - 1))
                            jitter = random.uniform(0, 0.5)
                            sleep_time = min(backoff_base + jitter, 60.0)
                            print(f"GPT hypothesis generation error on attempt {attempt}: {e} -- retrying after {sleep_time:.2f}s")
                            time.sleep(sleep_time)
                            continue
                        else:
                            print(f"Non-retriable GPT error during hypothesis generation: {e}")
                            break
                if not hypotheses_text:
                    print(f"No hypotheses generated for category '{category}'")
                    continue
            finally:
                try:
                    if acquired:
                        self._concurrent_semaphore.release()
                except Exception:
                    pass

            # Parse hypotheses: expect lines starting with number + dot, e.g. "1. Hypothesis text"
            hypotheses = []
            for line in hypotheses_text.splitlines():
                m = re.match(r"^\s*\d+\.\s*(.+)$", line)
                if m:
                    hypotheses.append(m.group(1).strip())
            if not hypotheses:
                # fallback: treat entire output as single hypothesis
                hypotheses = [hypotheses_text.strip()]

            # Limit to max_hypotheses
            hypotheses = hypotheses[:max_hypotheses]

            # Evaluate these hypotheses on the example (True/False)
            results = self.query_gpt_batch(hypotheses, inp_grid.tolist(), out_grid.tolist())

            # Score hypotheses with Solomonoff
            category_score = 0.0
            hypothesis_details = []
            for hyp, passed in zip(hypotheses, results):
                complexity = rule_complexity(hyp)
                score = calculate_solomonoff_score([passed], prior=1.0, complexity=complexity) if passed else 0.0
                hypothesis_details.append((hyp, passed, complexity, score))
                if passed:
                    category_score += score

            refined_scores[category] = category_score
            detailed_hypotheses[category] = hypothesis_details

        return refined_scores, detailed_hypotheses



    ####################################################################################
    # Modified run() method with two-stage category narrowing + refined ranking
    ####################################################################################
    def run(self, save_results=True, top_n_categories=3):
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

            # === Step 1: Narrow down plausible categories with predefined rules ===
            category_scores = {}
            detailed_results = {}

            for category in CATEGORIES:
                score, passed_rules = self.evaluate_category(task, category, ALL_RULES.get(category, []))
                category_scores[category] = score
                detailed_results[category] = passed_rules

            # Softmax normalize initial scores
            scores_arr = np.array(list(category_scores.values()))
            normalized_scores = softmax(scores_arr)
            normalized_category_scores = {cat: normalized_scores[i] for i, cat in enumerate(CATEGORIES)}

            # Sort categories by normalized initial score descending
            sorted_categories = sorted(normalized_category_scores.keys(),
                                       key=lambda c: normalized_category_scores[c],
                                       reverse=True)

            # === Step 2: For top N categories, generate & evaluate GPT hypotheses independently ===
            refined_scores, detailed_hypotheses = self.generate_and_evaluate_hypotheses_for_categories(
                task, sorted_categories, top_n=top_n_categories)

            # Softmax normalize refined scores
            if refined_scores:
                vals = np.array(list(refined_scores.values()))
                refined_norm = softmax(vals)
                refined_normalized_scores = {cat: refined_norm[i] for i, cat in enumerate(refined_scores.keys())}
                # Sort refined categories by score
                refined_sorted = sorted(refined_normalized_scores.keys(), key=lambda c: refined_normalized_scores[c], reverse=True)
                best_category = refined_sorted[0]
            else:
                refined_normalized_scores = {}
                best_category = sorted_categories[0] if sorted_categories else None

            # Save to task for output
            task['predicted_scores_initial'] = normalized_category_scores
            task['predicted_scores_refined'] = refined_normalized_scores
            task['predicted_categories_initial'] = [sorted_categories[0]] * len(task.get("train", [])) if sorted_categories else []
            task['predicted_categories_refined'] = [best_category] * len(task.get("train", [])) if best_category else []
            task['detailed_rule_results_initial'] = detailed_results
            task['detailed_hypotheses_refined'] = detailed_hypotheses

            if self.manual_results and task_name in self.manual_results:
                task['expected_category'] = self.manual_results[task_name]

            if save_results:
                output_path = self.output_folder / f"{task_path.stem}_evaluated.json"
                with open(output_path, "w") as outf:
                    json.dump(task, outf, indent=2)

            # Check correctness
            expected = task.get('expected_category')
            is_correct = expected == best_category
            return is_correct, task_name

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(process_task, (i, task)) for i, task in enumerate(tasks, 1)]
            for future in tqdm(as_completed(futures), total=total_tasks, desc="Processing tasks"):
                try:
                    correct, tname = future.result()
                    if correct:
                        correct_count += 1
                        correct_tasks.append(tname)
                except Exception as e:
                    print(f"Task processing error: {e}")

        print(f"\nSummary: {correct_count} / {total_tasks} tasks correctly categorized (refined step)")
        return correct_count, total_tasks, correct_tasks

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
        import matplotlib.pyplot as plt
        import numpy as np
        import json

        if not self.task_data:
            print("No tasks loaded. Run the engine first.")
            return

        task_names = list(self.task_data.keys())

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
                pairs = [(np.array(pair["input"]), np.array(pair["output"])) for pair in task["train"]]
            except Exception as e:
                print(f"Error loading train pairs for task {task_name}: {e}")
                continue

            pred_cats_initial = task.get("predicted_categories_initial", [])
            pred_cats_refined = task.get("predicted_categories_refined", [])

            scores_initial = task.get("predicted_scores_initial", {})
            scores_refined = task.get("predicted_scores_refined", {})

            expected_category = task.get("expected_category", None)

            compare_multiple_pairs(
                pairs,
                task_id=task_name,
                predicted_categories=pred_cats_refined or pred_cats_initial,
                expected_category=expected_category
            )

            print(f"\nExpected category: {expected_category if expected_category else 'N/A'}")
            print(f"Predicted categories initial: {pred_cats_initial if pred_cats_initial else 'N/A'}")
            print(f"Predicted categories refined: {pred_cats_refined if pred_cats_refined else 'N/A'}")

            # Print predicted scores clearly
            if scores_initial:
                print("\nInitial predicted scores:")
                for cat, score in scores_initial.items():
                    print(f"  {cat}: {score:.6f}")
                plot_solomonoff_scores(scores_initial, title=f"{task_name} - Initial Predicted Scores")

            if scores_refined:
                print("\nRefined predicted scores:")
                for cat, score in scores_refined.items():
                    print(f"  {cat}: {score:.6f}")
                plot_solomonoff_scores(scores_refined, title=f"{task_name} - Refined Predicted Scores")

            # Print detailed hypotheses initial and refined
            for version in ["initial", "refined"]:
                detailed_hypotheses = task.get(f"detailed_hypotheses_{version}", {})
                if detailed_hypotheses:
                    print(f"\nDetailed Hypotheses ({version}):")
                    for category, hypotheses in detailed_hypotheses.items():
                        print(f"\nCategory: {category}")
                        for hypothesis in hypotheses:
                            desc = hypothesis[0]
                            passed = hypothesis[1]
                            idx = hypothesis[2]
                            score = hypothesis[3]
                            status = "✅" if passed else "❌"
                            print(f" - {status} [{idx}] (score={score:.3f}): {desc}")
                else:
                    print(f"No detailed hypotheses found for {version}.\n")

            # Print detailed rule results for initial and refined
            for version in ["initial", "refined"]:
                detailed_rules = task.get(f"detailed_rule_results_{version}", {})
                if detailed_rules:
                    print(f"\nPassing Rules ({version}):")
                    found_any = False
                    for category, rules_results in detailed_rules.items():
                        passing_rules = [desc for desc, passed in rules_results.items() if passed]
                        if passing_rules:
                            found_any = True
                            print(f"\nCategory: {category}")
                            for rule_desc in passing_rules:
                                print(f" - ✅ {rule_desc}")
                    if not found_any:
                        print(f"No passing rules found in {version} detailed results.")
                else:
                    print(f"No detailed rule results found for {version}.")

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
                plt.tight_layout()
                plt.close(fig)  # Close figure after showing or to free memory if showing later

    def load_tasks(self):
        self.task_data = {}
        json_files = list(self.data_folder.glob("*.json"))
        if not json_files:
            print(f"No JSON task files found in {self.data_folder}")
            return

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    self.task_data[json_file.name] = data
            except Exception as e:
                print(f"Failed to load {json_file.name}: {e}")

        print(f"Loaded {len(self.task_data)} tasks from {self.data_folder}")

        # Define evaluated folder (change to your actual path)
        self.evaluated_folder = Path(r"generated data\Test #1 10.08")

        correct_count = 0
        total_count = 0

        for task_name in self.task_data.keys():
            evaluated_filename = task_name.replace(".json", "_evaluated.json")
            evaluated_path = self.evaluated_folder / evaluated_filename

            if evaluated_path.exists():
                total_count += 1
                try:
                    with open(evaluated_path, 'r') as f_eval:
                        evaluated_task = json.load(f_eval)
                except Exception as e:
                    print(f"Failed to load evaluated file {evaluated_filename}: {e}")
                    continue

                expected = evaluated_task.get("expected_category", None)
                preds = evaluated_task.get("predicted_categories_refined") or evaluated_task.get("predicted_categories_initial") or []

                if expected and preds and expected in preds:
                    correct_count += 1
            else:
                print(f"Warning: Evaluated file not found for {task_name}")

        print(f"Tasks correct (based on _evaluated files): {correct_count} / {total_count}")