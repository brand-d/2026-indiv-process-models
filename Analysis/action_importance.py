''' Script giving an overview about the percentages in which a certain Action was selected (by number of observations).
Cases with First-Order logic are ignored.
Usage to create the percentages in the Paper (Figure 1):
python action_importance.py Results/pipeline_fit.json
'''
import json
import sys

data = None
if len(sys.argv) != 2:
    print("Usage: python action_importance.py <file.json>")
    exit()

with open(sys.argv[1], "r") as file:
    data = json.load(file)

accumulation = {}
total = 0
for participant, params in data.items():
    if params["use_fol"]:
        continue
    total += 1

    for param_name, actions in params.items():
        if param_name == "use_fol":
            continue
        if param_name not in accumulation:
            accumulation[param_name] = {}
        if isinstance(actions, list):
            for action in actions:
                if action not in accumulation[param_name]:
                    accumulation[param_name][action] = 0
                accumulation[param_name][action] += 1
        else:
            if actions not in accumulation[param_name]:
                accumulation[param_name][actions] = 0
            accumulation[param_name][actions] += 1

for category, numbers in accumulation.items():
    print("Category:", category)
    sorted_numbers = sorted(numbers.items(), key=lambda x: x[1], reverse=True)
    for action, count in sorted_numbers:
        print("    {}: {} ({}%)".format(action, count, round((count / total) * 100, 1)))
