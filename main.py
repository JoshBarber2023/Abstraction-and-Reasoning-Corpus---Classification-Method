from rule_engine import HybridRuleEngine
import numpy as np
from pathlib import Path
import os

if __name__ == "__main__":
    # Set your OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY", None)
    
    if not openai_api_key:
        print("⚠️ Warning: No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
        print("The system needs AI to generate smart hypotheses and interpretations.")
        exit(1)
    
    # Paths  
    arc_folder = "./MINI-ARC/data/MiniARC"  # or your full ARC dataset
    data_folder = r"./generated data/Test #3 17.08"
    
    # Initialize the hybrid engine with controlled rate limits
    engine = HybridRuleEngine(
        arc_folder, 
        data_folder, 
        openai_api_key,
        max_concurrent_requests=2,  # Conservative to avoid rate limits
        rpm_limit=150               # Adjust based on your OpenAI plan
    )
    
    # Load all tasks
    engine.load_tasks()
    
    # Optional: Manual categorization for validation
    # Uncomment this to create ground truth for accuracy measurement:
    # engine.manual_categorize()
    
    # Run the hybrid AI system (now generates interpretations automatically!)
    print("🚀 Running Hybrid AI Rule Analysis System with Problem Interpretations...")
    engine.run(save_results=True)
    
    # NEW: Export clean interpretations summary for easy reference
    print("\n📝 Generating clean interpretations summary...")
    engine.export_interpretations_summary()
    
    # Get performance summary
    print("\n📊 Performance Analysis:")
    engine.get_performance_summary()
    
    # Analyze errors to understand what needs improvement
    print("\n❌ Error Analysis:")
    engine.analyze_errors()
    
    # NEW: Quick overview of all interpretations
    print("\n🧠 Problem Interpretations Overview:")
    engine.view_interpretations()
    
    # View detailed analysis of first few tasks (now includes interpretations!)
    print("\n🔍 Detailed analysis with AI interpretations:")
    
    # Ask user what they want to see
    print("\nChoose analysis mode:")
    print("1. View first task only (detailed)")
    print("2. View first 3 tasks (detailed)")  
    print("3. View specific task by name")
    print("4. Skip detailed view")
    
    try:
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            engine.View(0)  # First task with full interpretation
        elif choice == "2":
            engine.View(show_all=True, max_tasks=3)  # First 3 tasks
        elif choice == "3":
            task_name = input("Enter task name (e.g., 'task_001.json'): ").strip()
            engine.View(task_name)
        elif choice == "4":
            print("Skipping detailed view.")
        else:
            print("Invalid choice, showing first task:")
            engine.View(0)
            
    except KeyboardInterrupt:
        print("\nSkipping detailed analysis.")
    
    # NEW: Show specific interpretations if user wants
    print("\n💡 Want to see specific problem interpretations?")
    try:
        while True:
            task_input = input("Enter task name to see interpretation (or 'q' to quit): ").strip()
            if task_input.lower() in ['q', 'quit', 'exit']:
                break
            elif task_input:
                engine.view_interpretations(task_input)
            else:
                break
    except KeyboardInterrupt:
        pass
    
    print("\n✅ Analysis complete! Check the generated files:")
    print(f"  📊 Results: {data_folder}/evaluated_scores.json")
    print(f"  🧠 Hypotheses: {data_folder}/generated_hypotheses.json") 
    print(f"  🤖 AI Interpretations: {data_folder}/problem_interpretations.json")
    print(f"  📝 Clean Summary: {data_folder}/interpretations_summary.json")
    print(f"  📁 Individual tasks: {data_folder}/*_evaluated.json")
    
    print(f"\n🎯 The 'interpretations_summary.json' file contains easy-to-read")
    print(f"    explanations of what each problem is doing!")
    
    # Show sample interpretation
    try:
        interpretations_path = Path(data_folder) / "interpretations_summary.json"
        if interpretations_path.exists():
            import json
            with open(interpretations_path) as f:
                summary = json.load(f)
            
            if summary.get("problems"):
                sample_task = list(summary["problems"].keys())[0]
                sample_data = summary["problems"][sample_task]
                
                print(f"\n📖 Sample interpretation for {sample_task}:")
                print(f"   What it does: {sample_data['what_it_does']}")
                print(f"   Confidence: {sample_data['confidence']}")
                print(f"   Category: {sample_data['category']}")
                print(f"   Success rate: {sample_data['success_rate']}")
                
    except Exception as e:
        print(f"Could not show sample interpretation: {e}")