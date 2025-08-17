from rule_engine import RuleEngine
import numpy as np
from pathlib import Path
import os

if __name__ == "__main__":
    # Set your OpenAI API key (either here or as environment variable)
    openai_api_key = os.getenv("OPENAI_API_KEY", None)
    # Alternatively, set it directly: openai_api_key = "your-api-key-here"
    
    if not openai_api_key:
        print("⚠️ Warning: No OpenAI API key found. Set OPENAI_API_KEY environment variable or pass it directly.")
        print("The engine will still work but won't generate AI hypotheses.")
    
    # Paths
    arc_folder = "./MINI-ARC/data/MiniARC"  # or "./Training Sets/NORMAL-ARC/Training"
    data_folder = r"./generated data/Test #7 13.08"
    
    # Initialize the smart engine
    engine = RuleEngine(arc_folder, data_folder, openai_api_key)
    
    # Load all tasks
    engine.load_tasks()
    
    # Optional: Manual categorization for validation
    # engine.manual_categorize()
    
    # Run optimized AI-powered analysis with cost control
    print("🚀 Running optimized AI-powered rule analysis...")
    
    # Cost control options:
    # Option 1: Use AI selectively (recommended)
    engine.run(save_results=True, use_ai_hypotheses=True, max_ai_tasks=20)
    
    # Option 2: Templates only (fastest, cheapest)
    # engine.run(save_results=True, use_ai_hypotheses=False)
    
    # Option 3: AI for all tasks (most accurate but expensive)
    # engine.run(save_results=True, use_ai_hypotheses=True)
    
    # Visualize results
    print("\n🔍 Viewing analysis results...")
    engine.View()  # View first task
    
    # Analyze generated hypotheses
    print("\n📊 Analyzing generated hypotheses...")
    from hypothesis_analyzer import HypothesisAnalyzer
    
    analyzer = HypothesisAnalyzer(data_folder)
    
    # Compare overall predictions (show only errors for debugging)
    analyzer.compare_predictions(show_only_errors=True)
    
    # Analyze hypothesis generation patterns
    analyzer.analyze_hypothesis_patterns()
    
    # Find best performing hypotheses
    analyzer.find_best_hypotheses(top_n=5)
    
    # You can also analyze specific tasks:
    # analyzer.analyze_task_hypotheses("your_task.json", show_details=True)
    
    # You can also view specific tasks:
    # engine.View(0)  # View first task by index
    # engine.View("specific_task.json")  # View by filename