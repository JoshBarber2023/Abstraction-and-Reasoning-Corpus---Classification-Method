import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt

class ImprovedHypothesisAnalyzer:
    """Enhanced analyzer with better readability and insights"""
    
    def __init__(self, output_folder):
        self.output_folder = Path(output_folder)
    
    def analyze_task_hypotheses(self, task_name: str, show_details: bool = True):
        """Analyze hypotheses for a specific task with improved formatting"""
        
        task_file = self.output_folder / f"{task_name.replace('.json', '')}_evaluated.json"
        if not task_file.exists():
            print(f"❌ Task file not found: {task_file}")
            return
            
        with open(task_file) as f:
            task_data = json.load(f)
        
        hypotheses = task_data.get('generated_hypotheses', [])
        if not hypotheses:
            print(f"❌ No hypotheses found for: {task_name}")
            return
            
        print(f"\n🔬 HYPOTHESIS ANALYSIS: {task_name}")
        print("=" * 55)
        
        # Task prediction summary
        predicted = task_data.get('predicted_categories', ['Unknown'])[0]
        expected = task_data.get('expected_category', 'Unknown')
        match = "✅" if predicted == expected else "❌"
        
        print(f"**Result:** {predicted} (expected: {expected}) {match}")
        
        # Hypothesis overview
        passed_all = sum(1 for h in hypotheses if h.get('passed_all', False))
        avg_pass_rate = np.mean([h.get('pass_rate', 0) for h in hypotheses])
        
        print(f"**Hypotheses:** {len(hypotheses)} total | {passed_all} perfect | {avg_pass_rate:.2f} avg pass rate")
        
        # Category distribution
        by_category = {}
        for hyp in hypotheses:
            cat = hyp['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(hyp)
        
        print(f"\n📊 **Category Breakdown:**")
        for cat, cat_hyps in sorted(by_category.items()):
            perfect = sum(1 for h in cat_hyps if h.get('passed_all', False))
            avg_rate = np.mean([h.get('pass_rate', 0) for h in cat_hyps])
            print(f"  {cat:12}: {len(cat_hyps)} hyps | {perfect} perfect | {avg_rate:.2f} avg")
        
        if show_details:
            print(f"\n📋 **Detailed Hypothesis Performance:**")
            print("-" * 55)
            
            # Sort by performance
            sorted_hyps = sorted(hypotheses, key=lambda x: (-x.get('pass_rate', 0), -x.get('confidence', 0)))
            
            for i, hyp in enumerate(sorted_hyps):
                # Status indicators
                if hyp.get('passed_all', False):
                    status = "🎯 PERFECT"
                elif hyp.get('pass_rate', 0) >= 0.8:
                    status = "🟢 STRONG"
                elif hyp.get('pass_rate', 0) >= 0.5:
                    status = "🟡 PARTIAL"
                else:
                    status = "🔴 WEAK"
                
                print(f"\n{i+1}. [{hyp['category']}] {status}")
                print(f"   📝 {hyp['description']}")
                print(f"   📊 Pass rate: {hyp.get('pass_rate', 0):.2f} | Confidence: {hyp.get('confidence', 0):.2f}")
                print(f"   🔍 Evidence: {hyp.get('evidence', 'None provided')}")
                
                # Show test failures if any
                test_results = hyp.get('test_results', [])
                if test_results and not all(test_results):
                    failed = [j for j, r in enumerate(test_results) if not r]
                    print(f"   ❌ Failed on examples: {failed}")
    
    def compare_predictions(self, show_only_errors: bool = False):
        """Enhanced prediction comparison with insights"""
        
        scores_file = self.output_folder / "evaluated_scores.json"
        if not scores_file.exists():
            print("❌ No scores file found. Run the engine first.")
            return
            
        with open(scores_file) as f:
            all_scores = json.load(f)
        
        correct = 0
        total = 0
        errors = []
        category_stats = {}
        
        print("\n🎯 PREDICTION ANALYSIS")
        print("=" * 50)
        
        for task_name, score_data in all_scores.items():
            predicted = score_data.get('predicted_category')
            expected = score_data.get('expected_category')
            
            if expected is None:
                continue
                
            total += 1
            match = predicted == expected
            
            # Track category statistics
            if expected not in category_stats:
                category_stats[expected] = {'correct': 0, 'total': 0}
            category_stats[expected]['total'] += 1
            
            if match:
                correct += 1
                category_stats[expected]['correct'] += 1
            else:
                errors.append((task_name, predicted, expected, score_data.get('scores', {})))
            
            # Display based on settings
            if not show_only_errors or not match:
                status = "✅" if match else "❌"
                confidence = score_data.get('scores', {}).get(predicted, 0)
                print(f"{status} {task_name}: {predicted} (expected: {expected}) conf: {confidence:.2f}")
        
        # Overall statistics
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"\n📈 **Overall Accuracy:** {accuracy:.1f}% ({correct}/{total})")
        
        # Category-wise performance
        print(f"\n📊 **Category Performance:**")
        for cat, stats in sorted(category_stats.items()):
            cat_accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {cat:12}: {cat_accuracy:5.1f}% ({stats['correct']}/{stats['total']})")
        
        # Error analysis
        if errors:
            print(f"\n❌ **Error Patterns** ({len(errors)} errors):")
            error_patterns = {}
            
            for task_name, predicted, expected, scores in errors:
                pattern = f"{expected} → {predicted}"
                if pattern not in error_patterns:
                    error_patterns[pattern] = []
                error_patterns[pattern].append((task_name, scores))
            
            for pattern, cases in sorted(error_patterns.items(), key=lambda x: -len(x[1]))[:5]:
                print(f"\n  **{pattern}:** {len(cases)} cases")
                
                # Analyze confidence in these errors
                correct_cat = pattern.split(' → ')[0]
                wrong_cat = pattern.split(' → ')[1]
                
                avg_correct_conf = np.mean([scores.get(correct_cat, 0) for _, scores in cases])
                avg_wrong_conf = np.mean([scores.get(wrong_cat, 0) for _, scores in cases])
                
                print(f"    Avg confidence in correct: {avg_correct_conf:.3f}")
                print(f"    Avg confidence in predicted: {avg_wrong_conf:.3f}")
                
                # Show examples
                for task_name, _ in cases[:3]:
                    print(f"    - {task_name}")
                if len(cases) > 3:
                    print(f"    ... and {len(cases) - 3} more")
        
        return {"accuracy": accuracy, "errors": errors, "category_stats": category_stats}
    
    def analyze_hypothesis_quality(self):
        """Analyze the quality and patterns of generated hypotheses"""
        
        hypotheses_file = self.output_folder / "generated_hypotheses.json"
        if not hypotheses_file.exists():
            print("❌ No hypotheses file found.")
            return
            
        with open(hypotheses_file) as f:
            all_hypotheses = json.load(f)
        
        print("\n🧠 HYPOTHESIS QUALITY ANALYSIS")
        print("=" * 50)
        
        # Collect statistics
        total_hypotheses = 0
        quality_stats = {
            'perfect': 0,      # 100% pass rate
            'strong': 0,       # 80%+ pass rate  
            'moderate': 0,     # 50-79% pass rate
            'weak': 0,         # <50% pass rate
        }
        
        category_performance = {}
        source_performance = {}
        
        for task_name, hypotheses in all_hypotheses.items():
            total_hypotheses += len(hypotheses)
            
            for hyp in hypotheses:
                pass_rate = hyp.get('pass_rate', 0)
                category = hyp.get('category', 'Unknown')
                source = hyp.get('source', 'unknown')
                
                # Quality classification
                if pass_rate >= 1.0:
                    quality_stats['perfect'] += 1
                elif pass_rate >= 0.8:
                    quality_stats['strong'] += 1
                elif pass_rate >= 0.5:
                    quality_stats['moderate'] += 1
                else:
                    quality_stats['weak'] += 1
                
                # Category performance tracking
                if category not in category_performance:
                    category_performance[category] = []
                category_performance[category].append(pass_rate)
                
                # Source performance tracking
                if source not in source_performance:
                    source_performance[source] = []
                source_performance[source].append(pass_rate)
        
        print(f"**Total hypotheses analyzed:** {total_hypotheses}")
        print(f"**Average per task:** {total_hypotheses / len(all_hypotheses):.1f}")
        
        # Quality distribution
        print(f"\n📊 **Quality Distribution:**")
        for quality, count in quality_stats.items():
            percentage = (count / total_hypotheses * 100) if total_hypotheses > 0 else 0
            print(f"  {quality.capitalize():9}: {count:4d} ({percentage:5.1f}%)")
        
        # Category performance
        print(f"\n🎯 **Category Performance:**")
        for category, rates in sorted(category_performance.items()):
            avg_rate = np.mean(rates)
            perfect_count = sum(1 for r in rates if r >= 1.0)
            print(f"  {category:12}: {avg_rate:.3f} avg | {perfect_count}/{len(rates)} perfect")
        
        # Source analysis
        if source_performance:
            print(f"\n🤖 **Source Performance:**")
            for source, rates in sorted(source_performance.items()):
                avg_rate = np.mean(rates)
                count = len(rates)
                print(f"  {source:15}: {avg_rate:.3f} avg ({count} hypotheses)")
        
        return {
            "total": total_hypotheses,
            "quality_stats": quality_stats,
            "category_performance": category_performance
        }
    
    def find_best_and_worst_hypotheses(self, top_n: int = 5):
        """Find the best and worst performing hypotheses with insights"""
        
        hypotheses_file = self.output_folder / "generated_hypotheses.json"
        if not hypotheses_file.exists():
            print("❌ No hypotheses file found.")
            return
            
        with open(hypotheses_file) as f:
            all_hypotheses = json.load(f)
        
        # Collect all hypotheses with task context
        all_hyps_with_context = []
        for task_name, hypotheses in all_hypotheses.items():
            for hyp in hypotheses:
                hyp_with_context = {**hyp, 'task_name': task_name}
                all_hyps_with_context.append(hyp_with_context)
        
        if not all_hyps_with_context:
            print("❌ No hypotheses found.")
            return
        
        # Sort by performance
        sorted_hyps = sorted(all_hyps_with_context, 
                           key=lambda x: (-x.get('pass_rate', 0), -x.get('confidence', 0)))
        
        print(f"\n🏆 TOP {top_n} BEST HYPOTHESES")
        print("=" * 60)
        
        for i, hyp in enumerate(sorted_hyps[:top_n]):
            source_icon = {"ai_contextual": "🧠", "ai_corrected": "🔧", "fallback": "📋"}.get(
                hyp.get('source', '').split('_')[0], "🤖"
            )
            
            print(f"\n{i+1}. {source_icon} [{hyp['category']}] - {hyp['task_name']}")
            print(f"   📝 {hyp['description']}")
            print(f"   📊 Pass rate: {hyp.get('pass_rate', 0):.2f} | Confidence: {hyp.get('confidence', 0):.2f}")
            print(f"   🎯 Tests: {hyp.get('test_results', [])}")
        
        # Worst hypotheses for learning
        worst_hyps = sorted_hyps[-top_n:]
        
        print(f"\n💥 WORST {top_n} HYPOTHESES (for improvement insights)")
        print("=" * 60)
        
        for i, hyp in enumerate(worst_hyps):
            print(f"\n{i+1}. [{hyp['category']}] - {hyp['task_name']}")
            print(f"   📝 {hyp['description']}")
            print(f"   📊 Pass rate: {hyp.get('pass_rate', 0):.2f}")
            print(f"   ❌ Failed tests: {[j for j, r in enumerate(hyp.get('test_results', [])) if not r]}")
    
    def generate_improvement_report(self):
        """Generate actionable insights for improving the system"""
        
        print("\n🔧 SYSTEM IMPROVEMENT REPORT")
        print("=" * 50)
        
        # Analyze predictions
        pred_analysis = self.compare_predictions(show_only_errors=True)
        
        # Analyze hypothesis quality  
        hyp_analysis = self.analyze_hypothesis_quality()
        
        # Generate recommendations
        print(f"\n💡 **RECOMMENDATIONS:**")
        
        accuracy = pred_analysis.get("accuracy", 0)
        if accuracy < 70:
            print(f"  🎯 **Low accuracy ({accuracy:.1f}%):** Focus on hypothesis quality improvement")
            
        perfect_rate = hyp_analysis["quality_stats"]["perfect"] / hyp_analysis["total"] * 100
        if perfect_rate < 20:
            print(f"  🧠 **Few perfect hypotheses ({perfect_rate:.1f}%):** Improve validation step")
            
        weak_rate = hyp_analysis["quality_stats"]["weak"] / hyp_analysis["total"] * 100
        if weak_rate > 40:
            print(f"  ❌ **Too many weak hypotheses ({weak_rate:.1f}%):** Strengthen generation")
        
        # Category-specific recommendations
        worst_categories = []
        for cat, stats in pred_analysis.get("category_stats", {}).items():
            if stats['total'] >= 3:  # Only for categories with enough samples
                cat_accuracy = stats['correct'] / stats['total'] * 100
                if cat_accuracy < 60:
                    worst_categories.append((cat, cat_accuracy))
        
        if worst_categories:
            print(f"  📂 **Problematic categories:**")
            for cat, acc in sorted(worst_categories, key=lambda x: x[1]):
                print(f"     - {cat}: {acc:.1f}% accuracy - needs better heuristics")
        
        # Cost optimization suggestions
        total_hyps = hyp_analysis["total"]
        avg_per_task = total_hyps / len(self.list_analyzed_tasks())
        
        if avg_per_task > 10:
            print(f"  💰 **High hypothesis count ({avg_per_task:.1f}/task):** Consider reducing for cost")
        
        if perfect_rate > 30:
            print(f"  ⚡ **Good hypothesis quality:** Can reduce count for efficiency")
            
        return {
            "accuracy": accuracy,
            "perfect_rate": perfect_rate,
            "weak_rate": weak_rate,
            "worst_categories": worst_categories
        }
    
    def list_analyzed_tasks(self):
        """Get list of analyzed tasks"""
        try:
            hypotheses_file = self.output_folder / "generated_hypotheses.json"
            if hypotheses_file.exists():
                with open(hypotheses_file) as f:
                    data = json.load(f)
                return list(data.keys())
            return []
        except:
            return []
    
    def quick_summary(self):
        """Provide a quick summary of system performance"""
        
        print("\n⚡ QUICK SUMMARY")
        print("=" * 30)
        
        try:
            # Load key metrics
            scores_file = self.output_folder / "evaluated_scores.json"
            if scores_file.exists():
                with open(scores_file) as f:
                    scores = json.load(f)
                
                total_tasks = len(scores)
                with_expected = sum(1 for s in scores.values() if s.get('expected_category'))
                correct = sum(1 for s in scores.values() 
                             if s.get('expected_category') and s.get('predicted_category') == s.get('expected_category'))
                
                accuracy = (correct / with_expected * 100) if with_expected > 0 else 0
                
                print(f"📊 Tasks analyzed: {total_tasks}")
                print(f"🎯 Accuracy: {accuracy:.1f}% ({correct}/{with_expected})")
            
            # Hypothesis stats
            hypotheses_file = self.output_folder / "generated_hypotheses.json"
            if hypotheses_file.exists():
                with open(hypotheses_file) as f:
                    hyps = json.load(f)
                
                total_hyps = sum(len(h) for h in hyps.values())
                perfect_hyps = sum(sum(1 for hyp in h if hyp.get('passed_all', False)) for h in hyps.values())
                perfect_rate = (perfect_hyps / total_hyps * 100) if total_hyps > 0 else 0
                
                print(f"🧠 Hypotheses: {total_hyps} total | {perfect_hyps} perfect ({perfect_rate:.1f}%)")
            
            # Interpretations
            interp_file = self.output_folder / "problem_interpretations.json"
            if interp_file.exists():
                with open(interp_file) as f:
                    interps = json.load(f)
                print(f"🤖 Interpretations: {len(interps)} generated")
            
            print(f"📁 Output folder: {self.output_folder}")
            
        except Exception as e:
            print(f"❌ Error loading summary: {str(e)[:50]}")


# Backwards compatibility
class HypothesisAnalyzer(ImprovedHypothesisAnalyzer):
    """Backwards compatible interface"""
    pass


def add_analyzer_to_main():
    """Integration code for main.py"""
    return '''
# Add this after engine analysis in your main.py:

from hypothesis_analyzer import ImprovedHypothesisAnalyzer

# Create analyzer
analyzer = ImprovedHypothesisAnalyzer(data_folder)

print("\\n🔬 DETAILED ANALYSIS")
print("=" * 40)

# Quick summary
analyzer.quick_summary()

# Prediction analysis
analyzer.compare_predictions(show_only_errors=False)

# Hypothesis quality
analyzer.analyze_hypothesis_quality() 

# Best/worst examples
analyzer.find_best_and_worst_hypotheses(top_n=3)

# Improvement recommendations
analyzer.generate_improvement_report()
'''