from rule_engine import HybridRuleEngine
import numpy as np
from pathlib import Path
import os
import time
import json

def generate_interpretations_only():
    """Standalone function to generate interpretations from existing data"""
    
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

def interactive_analysis_menu(engine):
    """Enhanced interactive analysis with more options"""
    try:
        # Get performance summary
        print("\n📊 Performance Analysis:")
        engine.get_performance_summary()
        
        # NEW: Quick overview of all interpretations
        print("\n🧠 Problem Interpretations Overview:")
        engine.view_interpretations()
        
        while True:
            # Main analysis menu
            print("\n🔍 Choose analysis mode:")
            print("1. View first task only (detailed)")
            print("2. View all tasks (detailed)")  
            print("3. View specific task by name")
            print("4. View N tasks (specify number)")
            print("5. View interpretations for specific task")
            print("6. View all interpretations summary")
            print("7. Skip detailed view / Exit")
            
            choice = input("Enter choice (1-7): ").strip()
            
            if choice == "1":
                print("\n🔍 Viewing first task in detail...")
                engine.View(0)
                
            elif choice == "2":
                print("\n🔍 Viewing ALL tasks in detail...")
                print("⚠️ This will show every task with pauses between them.")
                confirm = input("Continue? (y/N): ").strip().lower()
                if confirm in ['y', 'yes']:
                    engine.View(show_all=True)
                else:
                    print("Cancelled.")
                    
            elif choice == "3":
                task_name = input("Enter task name (e.g., 'task_001.json'): ").strip()
                if task_name:
                    print(f"\n🔍 Viewing task: {task_name}")
                    engine.View(task_name)
                else:
                    print("No task name entered.")
                    
            elif choice == "4":
                try:
                    num_tasks = int(input("How many tasks to view? ").strip())
                    if num_tasks > 0:
                        print(f"\n🔍 Viewing first {num_tasks} tasks...")
                        engine.View(show_all=True, max_tasks=num_tasks)
                    else:
                        print("Invalid number.")
                except ValueError:
                    print("Please enter a valid number.")
                    
            elif choice == "5":
                task_name = input("Enter task name for interpretation (e.g., 'task_001.json'): ").strip()
                if task_name:
                    engine.view_interpretations(task_name)
                else:
                    print("No task name entered.")
                    
            elif choice == "6":
                print("\n🧠 All interpretations overview:")
                engine.view_interpretations()
                
            elif choice == "7":
                print("Exiting analysis menu.")
                break
                
            else:
                print("Invalid choice. Please try again.")
                
            # Ask if user wants to continue
            if choice not in ["7"]:
                continue_choice = input("\nContinue with more analysis? (Y/n): ").strip().lower()
                if continue_choice in ['n', 'no']:
                    break
                    
    except KeyboardInterrupt:
        print("\nExiting interactive analysis.")

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
    
    # Check if existing data is available
    hypotheses_path = Path(data_folder) / "generated_hypotheses.json"
    has_existing_data = hypotheses_path.exists()
    
    print("🤖 Hybrid AI Rule Analysis System")
    print("=" * 40)
    
    # Initialize engine variable for later use
    engine = None
    
    if has_existing_data:
        print("✅ Found existing hypothesis data!")
        print("\nChoose mode:")
        print("1. Generate interpretations only (from existing data)")
        print("2. Run full analysis (regenerate everything)")
        print("3. Just view existing interpretations")
        print("4. View all tasks (detailed analysis)")
        print("5. Interactive analysis menu")
        print("6. Exit")
        
        try:
            mode = input("\nEnter choice (1-6): ").strip()
            
            if mode == "1":
                print("\n🚀 Generating interpretations from existing hypothesis data...")
                success = generate_interpretations_only()
                if not success:
                    exit(1)
                    
            elif mode == "2":
                print("\n🚀 Running full analysis...")
                # Initialize the hybrid engine with controlled rate limits
                engine = HybridRuleEngine(
                    arc_folder, 
                    data_folder, 
                    openai_api_key,
                    max_concurrent_requests=2,
                    rpm_limit=150
                )
                
                # Load all tasks
                engine.load_tasks()
                
                # Run the hybrid AI system
                engine.run(save_results=True)
                
                # Export interpretations
                engine.export_interpretations_summary()
                
            elif mode == "3":
                print("\n📚 Viewing existing interpretations...")
                engine = HybridRuleEngine(arc_folder, data_folder, openai_api_key)
                engine.view_interpretations()
                
            elif mode == "4":
                print("\n🔍 Viewing all tasks with detailed analysis...")
                engine = HybridRuleEngine(arc_folder, data_folder, openai_api_key)
                engine.load_tasks()  # Make sure tasks are loaded
                print("⚠️ This will show every task with pauses between them.")
                confirm = input("Continue? (y/N): ").strip().lower()
                if confirm in ['y', 'yes']:
                    engine.View(show_all=True)
                else:
                    print("Cancelled.")
                
            elif mode == "5":
                print("\n🎛️ Starting interactive analysis menu...")
                engine = HybridRuleEngine(arc_folder, data_folder, openai_api_key)
                engine.load_tasks()  # Make sure tasks are loaded
                interactive_analysis_menu(engine)
                
            elif mode == "6":
                print("Goodbye!")
                exit(0)
                
            else:
                print("Invalid choice. Using interpretation-only mode.")
                success = generate_interpretations_only()
                if not success:
                    exit(1)
                    
        except KeyboardInterrupt:
            print("\nExiting...")
            exit(0)
    else:
        print("❌ No existing hypothesis data found.")
        print("🚀 Running full analysis to generate data...")
        
        # Initialize the hybrid engine with controlled rate limits
        engine = HybridRuleEngine(
            arc_folder, 
            data_folder, 
            openai_api_key,
            max_concurrent_requests=2,
            rpm_limit=150
        )
        
        # Load all tasks
        engine.load_tasks()
        
        # Optional: Manual categorization for validation
        # Uncomment this to create ground truth for accuracy measurement:
        # engine.manual_categorize()
        
        # Run the hybrid AI system
        engine.run(save_results=True)
        
        # Export clean interpretations summary
        engine.export_interpretations_summary()
    
    # Interactive analysis section - only run if we have an engine initialized
    if engine is not None:
        print("\n🎯 Starting interactive analysis...")
        interactive_analysis_menu(engine)
    else:
        # Initialize engine for interactive analysis if not already done
        print("\n🎯 Initializing for interactive analysis...")
        engine = HybridRuleEngine(arc_folder, data_folder, openai_api_key)
        engine.load_tasks()
        interactive_analysis_menu(engine)
    
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