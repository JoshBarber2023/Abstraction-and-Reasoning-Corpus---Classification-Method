from rule_engine import RuleEngine
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    #arc_folder = "./Training Sets/NORMAL-ARC/Training"  # or "./MINI-ARC/data/MiniARC"
    arc_folder = "./MINI-ARC/data/MiniARC"
    data_folder = "./data" # where data is being saved to

    engine = RuleEngine(arc_folder, data_folder)

    engine.run()

    #engine.View("example_task.json")  # Replace with actual filename

    engine.View()  # To visualize the first task
