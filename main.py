from rule_engine import RuleEngine
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    #arc_folder = "./Training Sets/NORMAL-ARC/Training"  # or "./MINI-ARC/data/MiniARC"
    arc_folder = "./MINI-ARC/data/MiniARC"
    data_folder = r"./generated data/Test #2 17.05"  # raw string to avoid escape sequence issues


    engine = RuleEngine(arc_folder, data_folder)

    #engine.manual_categorize()

    engine.run()

    #engine.View("example_task.json")  # Replace with actual filename

    engine.View()  # To visualise the first task
