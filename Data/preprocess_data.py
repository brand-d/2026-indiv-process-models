''' Converts the syllogistic data from the cross-domain study (anonymized) to a dataset in the
format required by the cognitive model benchmarking tool CCOBRA.
It requires a prefix representation of tasks, separated by:
- ";": separates terms within a premise
- "/": separates premises
- "|": separates independent information (like in the response options/choices)

Therefore, "Some A are B, All B are C", would be:
"Some;A,B/All;B;C"
'''
import pandas as pd
import json
import ccobra
import os
from typing import Union, Dict, List
import domain_info as di

# Load the dataset
df = pd.read_csv(os.path.join("2025_crossdomain", "syllog_data.csv"))

# Collect the results
result : List[Dict[str, Union[str, int]]] = []

# Convert the entries line by line to CCOBRA form
for _, row in df.iterrows():
    task = row["task"]
    responses = json.loads(row["response"])
    terms = json.loads(row["terms"])

    # Generate the representations required for CCOBRA
    ccobra_task = ccobra.syllogistic.create_data_string_task(terms[0], terms[1], terms[2], task)
    choices = ccobra.syllogistic.create_data_string_choices(terms[0], terms[2])
    ccobra_responses = "|".join([ccobra.syllogistic.create_data_string_response(terms[0], terms[2], x) for x in responses])

    entry = {
        "id": row["id"],
        "sequence": row["seq"],
        "response_type": "multiple-choice",
        "domain": "syllogistic",
        "task": ccobra_task,
        "response": ccobra_responses,
        "choices": choices,
        "enc_task": task,
        "enc_responses": json.dumps(responses)
    }
    result.append(entry)

# Build dataframe
result_df = pd.DataFrame(result)

# Filter the data to only contain participants that completed all tasks in all domains
# This is useful to keep results stable when extending to the other domains
result_df = result_df[result_df["id"].isin(di.shared_vp)]

# Save to CSV
result_df.to_csv("2025_syllog_mc.csv", index=False)