from rule_engine import ImprovedHybridRuleEngine as HybridRuleEngine
import numpy as np
from pathlib import Path
import os
import time
import json
# Import the enhanced visualization functions
from utils.visualisation import visualize_all_tasks_comprehensive

def generate_interpretations_only():
    """Generate interpretations from existing hypothesis data with cost estimation"""
    
    openai_api_key = os.getenv("OPENAI_API_KEY", None)
    
    if not openai_api_key:
        print("⚠️ Warning: No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
        return False
    
    # Paths  
    arc_folder = "./MINI-ARC/data/MiniARC"
    data_folder = r"./generated data/Test #3 17.08"
    
    # Check existing data
    hypotheses_path = Path(data_folder) / "generated_hypotheses.json"
    if not hypotheses_path.exists():
        print("❌ No existing hypothesis data found.")
        return False
    
    # Load and estimate costs
    with open(hypotheses_path) as f:
        all_hypotheses = json.load(f)
    
    num_tasks_with_hypotheses = sum(1 for hyps in all_hypotheses.values() if hyps)
    estimated_cost = num_tasks_with_hypotheses * 0.002  # ~$0.002 per interpretation
    
    print(f"📊 Cost estimate: ~${estimated_cost:.2f} for {num_tasks_with_hypotheses} interpretations")
    
    if estimated_cost > 1.0:
        confirm = input(f"Estimated cost is ${estimated_cost:.2f}. Continue? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("Cancelled.")
            return False
    
    # Initialize lightweight engine
    engine = HybridRuleEngine(
        arc_folder, 
        data_folder, 
        openai_api_key,
        max_concurrent_requests=1,  # Conservative for cost control
        rpm_limit=100,
        enable_validation=True
    )
    
    print("🚀 Generating AI interpretations from existing data...")
    start_time = time.time()
    
    # Generate interpretations
    interpretations_generated = 0
    
    for task_name, hypotheses in all_hypotheses.items():
        if not hypotheses:
            continue
            
        try:
            # Load task data for context
            task_file = Path(data_folder) / task_name.replace('.json', '_evaluated.json')
            if task_file.exists():
                with open(task_file) as f:
                    task_data = json.load(f)
            else:
                task_data = {}
            
            print(f"   🔄 {task_name}...")
            
            interpretation = engine.generate_contextual_interpretation(hypotheses, task_name, task_data)
            
            # Store interpretation
            engine.problem_interpretations[task_name] = {
                'interpretation': interpretation,
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'num_hypotheses': len(hypotheses),
                'best_pass_rate': max([h.get('pass_rate', 0) for h in hypotheses]) if hypotheses else 0,
                'best_category': max(hypotheses, key=lambda x: x.get('pass_rate', 0)).get('category', 'Unknown') if hypotheses else 'Unknown'
            }
            
            interpretations_generated += 1
            time.sleep(0.8)  # Rate limiting
            
        except Exception as e:
            print(f"   ❌ Error with {task_name}: {str(e)[:50]}")
            continue
    
    # Save results
    if interpretations_generated > 0:
        interpretations_file = engine.save_problem_interpretations()
        elapsed = time.time() - start_time
        actual_cost = engine.ai_calls_made * 0.002  # Rough estimate
        
        print(f"\n✅ Generated {interpretations_generated} interpretations in {elapsed:.1f}s")
        print(f"💰 Estimated actual cost: ~${actual_cost:.2f}")
        print(f"📁 Saved to: {interpretations_file}")
        
        return True
    else:
        print("❌ No interpretations generated.")
        return False

def interactive_analysis_menu(engine):
    """Enhanced interactive menu with visualize all option"""
    try:
        print("\n📊 Quick Performance Overview:")
        engine.get_performance_summary()
        
        while True:
            print("\nAnalysis Options:")
            print("1. View first task (detailed)")
            print("2. View specific task") 
            print("3. View all interpretations")
            print("4. View specific interpretation")
            print("5. Error analysis")
            print("6. VISUALIZE ALL (comprehensive slideshow)")
            print("7. VISUALIZE ALL (auto-mode)")
            print("8. Export visualizations")
            print("9. Exit")
            
            choice = input("Choice (1-9): ").strip()
            
            if choice == "1":
                print("\n🔍 First task analysis:")
                engine.View(0)
                
            elif choice == "2":
                task_name = input("Task name (e.g., '00d62c1b.json'): ").strip()
                if task_name:
                    engine.View(task_name)
                    
            elif choice == "3":
                engine.view_interpretations()
                
            elif choice == "4":
                task_name = input("Task name for interpretation: ").strip()
                if task_name:
                    engine.view_interpretations(task_name)
                    
            elif choice == "5":
                engine.analyze_errors()
                
            elif choice == "6":
                print("Starting comprehensive visualization slideshow...")
                print("   (Interactive mode - use controls to navigate)")
                
                max_tasks = input("Max tasks to show (Enter for all): ").strip()
                max_tasks = int(max_tasks) if max_tasks.isdigit() else None
                
                visualize_all_tasks_comprehensive(
                    engine, 
                    max_tasks=max_tasks, 
                    save_individual=False, 
                    auto_advance=False
                )
                
            elif choice == "7":
                print("Starting auto-advance visualization...")
                
                max_tasks = input("Max tasks to show (Enter for all): ").strip()
                max_tasks = int(max_tasks) if max_tasks.isdigit() else None
                
                delay = input("Delay between tasks in seconds (default 3): ").strip()
                delay = float(delay) if delay else 3.0
                
                print(f"Auto-advancing every {delay}s...")
                print("   Press Ctrl+C to stop")
                
                try:
                    visualize_all_tasks_comprehensive(
                        engine, 
                        max_tasks=max_tasks, 
                        save_individual=False, 
                        auto_advance=True
                    )
                except KeyboardInterrupt:
                    print("\n⏹️ Auto-visualization stopped.")
                    
            elif choice == "8":
                print("\n💾 Export comprehensive visualizations...")
                
                save_dir = input("Save directory (default: ./visualizations): ").strip()
                if not save_dir:
                    save_dir = "./visualizations"
                
                max_tasks = input("Max tasks to export (Enter for all): ").strip()
                max_tasks = int(max_tasks) if max_tasks.isdigit() else None
                
                print(f"📁 Exporting to: {save_dir}")
                
                visualize_all_tasks_comprehensive(
                    engine, 
                    max_tasks=max_tasks, 
                    save_individual=True, 
                    auto_advance=True,  # Don't wait for user input when saving
                    save_dir=save_dir
                )
                
                print(f"✅ Export complete! Check {save_dir} folder.")
                
            elif choice == "9":
                print("👋 Goodbye!")
                break
                
            else:
                print("Invalid choice. Try again.")
                
            # Quick continuation prompt
            if choice not in ["6", "7", "8", "9"]:
                cont = input("\nContinue analysis? (Y/n): ").strip().lower()
                if cont in ['n', 'no']:
                    break
                    
    except KeyboardInterrupt:
        print("\n👋 Exiting...")

if __name__ == "__main__":
    # Configuration
    openai_api_key = os.getenv("OPENAI_API_KEY", None)
    
    if not openai_api_key:
        print("❌ OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        print("   The system needs AI to generate accurate hypotheses and interpretations.")
        exit(1)
    
    # Paths  
    arc_folder = "./MINI-ARC/data/MiniARC"
    data_folder = Path("./generated data/Test #3 17.08").resolve()
    hypotheses_path = data_folder / "generated_hypotheses.json"
    scores_path = data_folder / "evaluated_scores.json"

    print("Scores Path", scores_path) # generated data\Test #3 17.08\generated_hypotheses.json
    print("Hypotheses Path", hypotheses_path)

    has_existing_data = hypotheses_path.exists()
    has_results = scores_path.exists()
    
    print("🤖 Improved ARC Task Classifier")
    print("=" * 35)
    print("✨ Features: Smart hypotheses, contextual analysis, cost optimization")
    print("🎬 NEW: Comprehensive visualization with interpretations!")
    
    # Initialize engine for later use
    engine = None
    
    if has_existing_data:
        print("✅ Found existing analysis data!")
        print("\nQuick options:")
        print("1. 🧠 Generate interpretations only (cheap)")
        print("2. 🔄 Re-run full analysis (expensive)")
        print("3. 📊 Interactive analysis (existing data)")
        print("4. 🎬 VISUALIZE ALL (slideshow mode)")
        print("5. 🏃 Exit")
        
        try:
            mode = input("\nChoice (1-5): ").strip()
            
            if mode == "1":
                print("\n🧠 Generating interpretations from existing data...")
                success = generate_interpretations_only()
                if not success:
                    exit(1)
                    
                # Load engine for analysis
                engine = HybridRuleEngine(arc_folder, data_folder, openai_api_key)
                engine.load_tasks()
                
            elif mode == "2":
                print("\n🔄 Running full analysis...")
                
                # Cost warning for full analysis
                num_tasks = len(list(Path(arc_folder).glob("*.json")))
                estimated_full_cost = num_tasks * 0.05  # Conservative estimate
                
                print(f"⚠️  Full analysis estimated cost: ~${estimated_full_cost:.2f}")
                print("   This will regenerate all hypotheses and interpretations.")
                
                if estimated_full_cost > 5.0:
                    confirm = input(f"Estimated cost ${estimated_full_cost:.2f}. Continue? (y/N): ").strip().lower()
                    if confirm not in ['y', 'yes']:
                        print("Cancelled. Try option 1 for cheaper interpretation-only mode.")
                        exit(0)
                
                engine = HybridRuleEngine(
                    arc_folder, 
                    data_folder, 
                    openai_api_key,
                    max_concurrent_requests=2,  # Controlled for cost
                    rpm_limit=120
                )
                
                engine.load_tasks()
                engine.run(save_results=True)
                
            elif mode == "3":
                print("\n📊 Loading existing data for analysis...")
                engine = HybridRuleEngine(arc_folder, data_folder, openai_api_key)
                engine.load_tasks()
                
            elif mode == "4":
                print("\n🎬 Loading data for visualization...")
                engine = HybridRuleEngine(arc_folder, data_folder, openai_api_key)
                engine.load_tasks()
                
                print("🎭 Visualization Mode Selection:")
                print("1. Interactive slideshow (manual navigation)")
                print("2. Auto-advance slideshow")
                print("3. Export all as images")
                
                viz_mode = input("Visualization mode (1-3): ").strip()
                
                max_tasks = input("Max tasks to visualize (Enter for all): ").strip()
                max_tasks = int(max_tasks) if max_tasks.isdigit() else None
                
                if viz_mode == "1":
                    print("🎮 Starting interactive slideshow...")
                    visualize_all_tasks_comprehensive(
                        engine, 
                        max_tasks=max_tasks, 
                        save_individual=False, 
                        auto_advance=False
                    )
                    
                elif viz_mode == "2":
                    delay = input("Delay between tasks in seconds (default 2): ").strip()
                    delay = float(delay) if delay else 2.0
                    
                    print(f"⏱️ Auto-advancing every {delay}s... (Ctrl+C to stop)")
                    try:
                        import time as time_module
                        
                        # Modified version with timing
                        task_names = list(engine.task_data.keys())
                        if max_tasks:
                            task_names = task_names[:max_tasks]
                            
                        for i, task_name in enumerate(task_names):
                            print(f"\nAuto-showing task {i+1}/{len(task_names)}: {task_name}")
                            
                            # Quick visualization without interaction
                            visualize_all_tasks_comprehensive(
                                engine, 
                                max_tasks=1, 
                                save_individual=False, 
                                auto_advance=True
                            )
                            
                            if i < len(task_names) - 1:  # Don't wait after last task
                                time_module.sleep(delay)
                                
                    except KeyboardInterrupt:
                        print("\nAuto-visualization stopped.")
                        
                elif viz_mode == "3":
                    save_dir = input("Save directory (default: ./comprehensive_visualizations): ").strip()
                    if not save_dir:
                        save_dir = "./comprehensive_visualizations"
                    
                    print(f"Exporting comprehensive visualizations to: {save_dir}")
                    
                    visualize_all_tasks_comprehensive(
                        engine, 
                        max_tasks=max_tasks, 
                        save_individual=True, 
                        auto_advance=True,
                        save_dir=save_dir
                    )
                    
                    print(f"Export complete! Check {save_dir} folder.")
                
                # Skip regular interactive menu for visualization-only mode
                print("\n🎬 Visualization complete!")
                exit(0)
                
            elif mode == "5":
                print("👋 Goodbye!")
                exit(0)
                
            else:
                print("Invalid choice. Defaulting to interpretation-only mode.")
                success = generate_interpretations_only()
                if not success:
                    exit(1)
                engine = HybridRuleEngine(arc_folder, data_folder, openai_api_key)
                engine.load_tasks()
                
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            exit(0)
    
    else:
        print("❌ No existing data found.")
        print("🚀 Running initial full analysis...")
        
        # Cost estimate for new analysis
        try:
            num_tasks = len(list(Path(arc_folder).glob("*.json")))
            estimated_cost = num_tasks * 0.05
            
            print(f"📊 Estimated cost: ~${estimated_cost:.2f} for {num_tasks} tasks")
            print("   (Includes hypothesis generation + interpretations)")
            
            if estimated_cost > 3.0:
                confirm = input(f"Estimated cost is ${estimated_cost:.2f}. Continue? (y/N): ").strip().lower()
                if confirm not in ['y', 'yes']:
                    print("Cancelled. You can run this later when ready.")
                    exit(0)
        
        except:
            print("⚠️ Could not estimate costs. Proceeding with analysis.")
        
        # Initialize and run
        engine = HybridRuleEngine(
            arc_folder, 
            data_folder, 
            openai_api_key,
            max_concurrent_requests=2,
            rpm_limit=120
        )
        
        engine.load_tasks()
        
        # Optional: Manual categorization
        manual_categorize = input("\nCreate manual categorizations for accuracy measurement? (y/N): ").strip().lower()
        if manual_categorize in ['y', 'yes']:
            print("🖱️ Starting manual categorization interface...")
            engine.manual_categorize()
        
        # Run analysis
        engine.run(save_results=True)
    
    # Interactive analysis (common for all paths)
    if engine is not None:
        print("\n🎯 Starting interactive analysis...")
        interactive_analysis_menu(engine)
    else:
        print("❌ Engine not initialized for analysis.")
    
    # Final summary
    print("\n✅ Session complete! Generated files:")
    
    output_files = [
        (Path(data_folder) / "evaluated_scores.json", "📊 Category predictions"),
        (Path(data_folder) / "generated_hypotheses.json", "🧠 AI hypotheses"),
        (Path(data_folder) / "problem_interpretations.json", "🤖 Detailed interpretations"),
        (Path(data_folder) / "interpretations_summary.json", "📝 Clean summaries")
    ]
    
    for file_path, description in output_files:
        if file_path.exists():
            print(f"  ✅ {description}: {file_path.name}")
        else:
            print(f"  ❌ {description}: Not generated")
    
    # Show sample interpretation
    summary_path = Path(data_folder) / "interpretations_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            
            if summary.get("problems"):
                sample_task = list(summary["problems"].keys())[0]
                sample_data = summary["problems"][sample_task]
                
                print(f"\n📖 Sample interpretation for {sample_task}:")
                print(f"   What it does: {sample_data['what_it_does'][:100]}...")
                print(f"   Category: {sample_data['category']} ({sample_data['confidence']} confidence)")
                
        except Exception as e:
            pass
    
    print(f"\n💡 Key file: '{data_folder}/interpretations_summary.json' contains")
    print(f"    human-readable explanations of what each problem does!")
    
    if engine and hasattr(engine, 'ai_calls_made'):
        estimated_cost = engine.ai_calls_made * 0.002
        print(f"\n💰 Session cost: ~${estimated_cost:.2f} ({engine.ai_calls_made} API calls)")
    
    print(f"\n🎬 TIP: Use option 6 or 7 in the menu for comprehensive visualizations!")
    print(f"       These show grids + interpretations + analysis in one view.")