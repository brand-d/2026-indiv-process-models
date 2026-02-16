''' Creates a solvability analysis, testing every task/observation pair individually.
This reflects the local analysis from the paper.
To replicate the paper, call:
python solvability.py ../Data/2025_syllog_mc.csv

'''
import ProcessModel
import pandas as pd
import json
import sys
import os

if len(sys.argv) != 2:
    print("usage: python solvability.py [dataset]")
    exit()
file = sys.argv[1]

df = pd.read_csv(file)

unsolvables = {}
total = 0
correct = 0
common_solutions = {}

for _it, row in df.iterrows():
    if _it % 100 == 0:
        print(_it, "/", len(df))

    task = row["enc_task"]
    resp = json.loads(row["enc_responses"])
    resp = ["NVC" if x == "nvc" else x for x in resp]
    
    solutions = ProcessModel.plan(task, resp, use_fol=False, verbose=False)
    if solutions:
        correct += 1

        for solution in solutions:
            sol_str = ";".join([x.get_name() for x in solution["path"]])
            if sol_str not in common_solutions:
                common_solutions[sol_str] = 0
            common_solutions[sol_str] += 1
    else:
        key = "{}_[{}]".format(task, ",".join(sorted(resp)))
        if key not in unsolvables:
            unsolvables[key] = 0
        unsolvables[key] += 1
    total += 1


with open(os.path.join("solvability", "{}_log.txt".format(os.path.basename(file))), "w") as out:
    print("Total solved: {} / {}".format(correct, total), file=out)
    print("    -->", correct / total, file=out)
    print(file=out)
    print("Unsolvables:", file=out)
    sorted_dict = {key: val for key, val in sorted(unsolvables.items(), key=lambda item: item[1], reverse=True)}
    for key, val in sorted_dict.items():
        print(key, val, file=out)

    print(file=out)
    print("Common Solutions", file=out)
    sorted_sols = {key: val for key, val in sorted(common_solutions.items(), key=lambda item: item[1], reverse=True)}
    for key, val in list(sorted_sols.items())[:100]:
        print(key, val, file=out)