import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt

class HypothesisAnalyzer:
    """Tool to analyze and debug generated hypotheses"""
    
    def __init__(self, output_folder):
        self.output_folder = Path(output_folder)
    
    def analyze_task_hypotheses(self, task_name: str, show_details: bool = True):
        """Analyze hypotheses for a specific task"""
        
        # Load the evaluated task file
        task_file = self.output_folder / f"{task_name.replace('.json', '')}_evaluated.json"
        if not task_file.exists():
            print(f"❌ Task file not found: {task_file}")
            return
            
        with open(task_file) as f:
            task_data = json.load(f)
        
        hypotheses = task_data.get('generated_hypotheses', [])
        if not hypotheses:
            print(f"❌ No hypotheses found for task: {task_name}")
            return
            
        print(f"\n🔍 HYPOTHESIS ANALYSIS: {task_name}")
        print("=" * 60)
        
        # Show task prediction vs expected
        predicted = task_data.get('predicted_categories', ['Unknown'])[0]
        expected = task_data.get('expected_category', 'Unknown')
        match = "✅" if predicted == expected else "❌"
        
        print(f"Predicted: {predicted} | Expected: {expected} {match}")
        print(f"Total hypotheses generated: {len(hypotheses)}")
        
        # Group by category
        by_category = {}
        for hyp in hypotheses:
            cat = hyp['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(hyp)
        
        print(f"\nHypotheses by category:")
        for cat, cat_hyps in by_category.items():
            avg_pass_rate = np.mean([h.get('pass_rate', 0) for h in cat_hyps])
            print(f"  {cat}: {len(cat_hyps)} hypotheses (avg pass rate: {avg_pass_rate:.2f})")
        
        if show_details:
            print(f"\n📋 DETAILED HYPOTHESIS BREAKDOWN:")
            print("-" * 60)
            
            # Sort by pass rate and score
            sorted_hyps = sorted(hypotheses, key=lambda x: (-x.get('pass_rate', 0), -x.get('solomonoff_score', 0)))
            
            for i, hyp in enumerate(sorted_hyps):
                source_icon = "🤖" if hyp.get('source') == 'ai' else "📋"
                pass_icon = "✅" if hyp.get('passed_all', False) else "❌"
                
                print(f"\n{i+1}. {source_icon} [{hyp['category']}] {pass_icon}")
                print(f"   Description: {hyp['description']}")
                print(f"   Pass Rate: {hyp.get('pass_rate', 0):.2f} | Score: {hyp.get('solomonoff_score', 0):.3f}")
                print(f"   Complexity: {hyp.get('complexity', 0)} | Confidence: {hyp.get('confidence', 0):.2f}")
                print(f"   Test Results: {hyp.get('test_results', [])}")
                
                # Show test code for failed hypotheses
                if not hyp.get('passed_all', False) and 'test_code' in hyp:
                    print(f"   Test Code Preview: {hyp['test_code'][:100]}...")
    
    def compare_predictions(self, show_only_errors: bool = False):
        """Compare predictions across all tasks"""
        
        scores_file = self.output_folder / "evaluated_scores.json"
        if not scores_file.exists():
            print("❌ Scores file not found. Run the engine first.")
            return
            
        with open(scores_file) as f:
            all_scores = json.load(f)
        
        correct = 0
        total = 0
        errors = []
        
        print("\n📊 PREDICTION COMPARISON")
        print("=" * 60)
        
        for task_name, score_data in all_scores.items():
            predicted = score_data.get('predicted_category')
            expected = score_data.get('expected_category')
            
            if expected is None:
                continue  # Skip tasks without manual categorization
                
            total += 1
            match = predicted == expected
            if match:
                correct += 1
            else:
                errors.append((task_name, predicted, expected))
            
            if not show_only_errors or not match:
                status = "✅" if match else "❌"
                print(f"{status} {task_name}: {predicted} (expected: {expected})")
        
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"\n🎯 Overall Accuracy: {accuracy:.1f}% ({correct}/{total})")
        
        if errors:
            print(f"\n❌ Error Analysis ({len(errors)} errors):")
            error_patterns = {}
            for task_name, predicted, expected in errors:
                pattern = f"{expected} → {predicted}"
                if pattern not in error_patterns:
                    error_patterns[pattern] = []
                error_patterns[pattern].append(task_name)
            
            for pattern, tasks in error_patterns.items():
                print(f"  {pattern}: {len(tasks)} tasks")
                for task in tasks[:3]:  # Show first 3 examples
                    print(f"    - {task}")
                if len(tasks) > 3:
                    print(f"    ... and {len(tasks) - 3} more")
    
    def analyze_hypothesis_patterns(self):
        """Analyze patterns in hypothesis generation"""
        
        hypotheses_file = self.output_folder / "generated_hypotheses.json"
        if not hypotheses_file.exists():
            print("❌ Hypotheses file not found.")
            return
            
        with open(hypotheses_file) as f:
            all_hypotheses = json.load(f)
        
        print("\n🧠 HYPOTHESIS PATTERN ANALYSIS")
        print("=" * 60)
        
        # Collect statistics
        total_hypotheses = 0
        by_source = {"ai": 0, "template": 0}
        by_category = {}
        by_success = {"passed_all": 0, "partial": 0, "failed": 0}
        
        for task_name, hypotheses in all_hypotheses.items():
            total_hypotheses += len(hypotheses)
            
            for hyp in hypotheses:
                # Source statistics
                source = hyp.get('source', 'unknown')
                if source in by_source:
                    by_source[source] += 1
                
                # Category statistics
                category = hyp.get('category', 'Unknown')
                by_category[category] = by_category.get(category, 0) + 1
                
                # Success statistics
                pass_rate = hyp.get('pass_rate', 0)
                if pass_rate == 1.0:
                    by_success["passed_all"] += 1
                elif pass_rate > 0:
                    by_success["partial"] += 1
                else:
                    by_success["failed"] += 1
        
        print(f"Total hypotheses generated: {total_hypotheses}")
        print(f"Average per task: {total_hypotheses / len(all_hypotheses):.1f}")
        
        print(f"\nBy Source:")
        for source, count in by_source.items():
            pct = count / total_hypotheses * 100
            print(f"  {source}: {count} ({pct:.1f}%)")
        
        print(f"\nBy Category:")
        for category, count in sorted(by_category.items(), key=lambda x: -x[1]):
            pct = count / total_hypotheses * 100
            print(f"  {category}: {count} ({pct:.1f}%)")
        
        print(f"\nBy Success Rate:")
        for success_type, count in by_success.items():
            pct = count / total_hypotheses * 100
            print(f"  {success_type}: {count} ({pct:.1f}%)")
    
    def find_best_hypotheses(self, top_n: int = 10):
        """Find the best performing hypotheses across all tasks"""
        
        hypotheses_file = self.output_folder / "generated_hypotheses.json"
        if not hypotheses_file.exists():
            print("❌ Hypotheses file not found.")
            return
            
        with open(hypotheses_file) as f:
            all_hypotheses = json.load(f)
        
        # Collect all hypotheses with task info
        all_hyps_with_task = []
        for task_name, hypotheses in all_hypotheses.items():
            for hyp in hypotheses:
                hyp_with_task = {**hyp, 'task_name': task_name}
                all_hyps_with_task.append(hyp_with_task)
        
        # Sort by pass rate, then by negative score (lower score is better)
        sorted_hyps = sorted(all_hyps_with_task, 
                           key=lambda x: (-x.get('pass_rate', 0), x.get('solomonoff_score', float('inf'))))
        
        print(f"\n🏆 TOP {top_n} BEST HYPOTHESES")
        print("=" * 60)
        
        for i, hyp in enumerate(sorted_hyps[:top_n]):
            source_icon = "🤖" if hyp.get('source') == 'ai' else "📋"
            print(f"\n{i+1}. {source_icon} [{hyp['category']}] - {hyp['task_name']}")
            print(f"   {hyp['description']}")
            print(f"   Pass Rate: {hyp.get('pass_rate', 0):.2f} | Score: {hyp.get('solomonoff_score', 0):.3f}")

# Usage example for the main script
def add_analysis_to_main():
    """Add this to your main.py after running the engine"""
    return '''
    # Add this after engine.View() in your main.py:
    
    # Analyze hypotheses
    from hypothesis_analyzer import HypothesisAnalyzer
    
    analyzer = HypothesisAnalyzer(data_folder)
    
    # Compare overall predictions
    analyzer.compare_predictions(show_only_errors=True)
    
    # Analyze hypothesis patterns
    analyzer.analyze_hypothesis_patterns()
    
    # Find best hypotheses
    analyzer.find_best_hypotheses(top_n=5)
    
    # Analyze specific task (replace with actual task name)
    # analyzer.analyze_task_hypotheses("your_task.json")
    '''