from rule_engine import RuleEngine
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    #arc_folder = "./Training Sets/NORMAL-ARC/Training"  # or "./MINI-ARC/data/MiniARC"
    arc_folder = "./MINI-ARC/data/MiniARC"
    data_folder = r"./generated data/Test #1 27.07"  # raw string to avoid escape sequence issues


    engine = RuleEngine(arc_folder, data_folder)

    #engine.manual_categorize()                                                       
 
    engine.run()

    #engine.View("example_task.json")  # Replace with actual filename

    engine.load_tasks() # loads the tasks before viewing

    engine.View()  # To visualise the first task

    # 51,406 Tokens per run ~0.02 cents
    # evaluate 4 full categories ~7 cents.
    # evaluate top 3 categories ~3 cents
    # Optimised such that it runs through all 'common sense' checks and only runs the GPT model to really crack down on the remaining categories.