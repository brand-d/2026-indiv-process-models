""" Loads the relevant information from all datasets.
It is then used by the recommender, the correlation and the metrics table generation.
However, for easier analyses, the metrics tables can also be used directly.

Taken from the (anonymized).
"""

import pandas as pd
import numpy as np
import ccobra
import json
import os

data_path = "2025_crossdomain"
selected_syl = ccobra.syllogistic.SYLLOGISMS

# Load datasets
syllogs_df = pd.read_csv(os.path.join(data_path, "syllog_data.csv"))
crt_df = pd.read_csv(os.path.join(data_path, "crt_data.csv"))
nfc_df = pd.read_csv(os.path.join(data_path, "nfc_data.csv"))
wason_df = pd.read_csv(os.path.join(data_path, "wason_data.csv"))
arrange_df = pd.read_csv(os.path.join(data_path, "spatial_arrangement_data.csv"))
corsi_df = pd.read_csv(os.path.join(data_path, "corsi_data.csv"))
syl_corr_df = pd.read_csv(os.path.join(data_path, "syl_correct.csv"))
verbal_df = pd.read_csv(os.path.join(data_path, "verbal_data.csv"))
mental_df = pd.read_csv(os.path.join(data_path, "mental_rotation_data.csv"))
cnd_df = pd.read_csv(os.path.join(data_path, "conditionals_data.csv"))
spatial_df = pd.read_csv(os.path.join(data_path, "spatial_rel_data.csv"))
carddir_df = pd.read_csv(os.path.join(data_path, "carddir_data.csv"))

# Determine VP that completed everything
shared_vp = set(crt_df["id"]).intersection(set(corsi_df["id"])).intersection(set(syl_corr_df["id"])) \
    .intersection(set(verbal_df["id"])).intersection(set(mental_df["id"])).intersection(set(cnd_df["id"])) \
    .intersection(set(spatial_df["id"])).intersection(set(carddir_df["id"])).intersection(set(nfc_df["id"])) \
    .intersection(set(arrange_df["id"])).intersection(set(wason_df["id"]).intersection(set(syllogs_df["id"])))
print("Number of VP:", len(shared_vp))

# -------------------------------------------------------------------------
# Syllogistic Reasoning
# -------------------------------------------------------------------------
syllogs_df["is_nvc"] = syllogs_df["response"].apply(lambda x: json.loads(x)[0] == "nvc")

syllog_participants = {}
for vp, p_df in syllogs_df.groupby("id"):
    responses = []
    for syl in selected_syl:
        resps = p_df[p_df["task"] == syl]["response"]
        if resps.empty:
            responses = []
            continue
            
        resps = json.loads(resps.values[0])
        responses.append(resps)
    if len(responses) == len(selected_syl):
        syllog_participants[vp] = responses

syl_corr_jacc = {}
syl_corr_set = {}
syl_corr_mae = {}
for vp, df in syl_corr_df.groupby("id"):
    if vp not in shared_vp:
        continue
    syl_corr_jacc[vp] = [df["jacc"].mean()]
    syl_corr_set[vp] = [df["set_corr"].mean()]
    syl_corr_mae[vp] = [df["mae"].mean()]

syl_some_not = {}
for vp, df in syllogs_df.groupby("id"):
    if vp not in shared_vp:
        continue
    syl_some_not[vp] = [df["some_all"].mean()]

syl_nvc = {}
for vp, df in syllogs_df.groupby("id"):
    if vp not in shared_vp:
        continue
    syl_nvc[vp] = [df["is_nvc"].mean()]

# -------------------------------------------------------------------------
# Spatial Arrangement
# -------------------------------------------------------------------------
arrange_correct = {}
arrange_pmm = {}
arrange_model_correct = {}
for vp, df in arrange_df.groupby("id"):
    if vp not in shared_vp:
        continue
    arrange_correct[vp] = [df["correct"].mean()]
    arrange_pmm[vp] = [df["pmm_sorting"].mean()]
    arrange_model_correct[vp] = [df["correct_sorting"].mean()]


# -------------------------------------------------------------------------
# Corsi
# -------------------------------------------------------------------------
corsi_info = {}
for vp, df in corsi_df.groupby("id"):
    if vp not in shared_vp:
        continue
    corsi_info[vp] = [df["max_corsi"].mean()]

# -------------------------------------------------------------------------
# NFC
# -------------------------------------------------------------------------
nfc_info = {}
for vp, df in nfc_df.groupby("id"):
    if vp not in shared_vp:
        continue
    nfc_info[vp] = [df["NFC"].mean()]

# -------------------------------------------------------------------------
# Wason
# -------------------------------------------------------------------------
wason_info = {}
wason_subset_info = {}
for vp, df in wason_df.groupby("id"):
    if vp not in shared_vp:
        continue
    wason_info[vp] = [df["jacc"].mean()]
    wason_subset_info[vp] = [df["correct_subset"].mean()]


# -------------------------------------------------------------------------
# CRT
# -------------------------------------------------------------------------
crt_info = {}
for vp, df in crt_df.groupby("id"):
    if vp not in shared_vp:
        continue
    crt_info[vp] = [df["correct"].mean()]

# -------------------------------------------------------------------------
# Verbal
# -------------------------------------------------------------------------
verbal_info = {}
for vp, df in verbal_df.groupby("id"):
    if vp not in shared_vp:
        continue
    verbal_info[vp] = [df["correct"].mean()]

# -------------------------------------------------------------------------
# Mental Rot
# -------------------------------------------------------------------------

mental_into = {}
for vp, df in mental_df.groupby("id"):
    if vp not in shared_vp:
        continue
    mental_into[vp] = [df["mr_score"].mean()]

# -------------------------------------------------------------------------
# Conditionals
# -------------------------------------------------------------------------
def get_cond_desc(elem):
    name = "{}{}".format(elem["task"], " (CF)" if elem["counterfactual"] else "")
    return name

def get_cond_solution(elem):
    if elem == "MP":
        return "positive"
    if elem == "MT":
        return "negative"
    
    return "nvc"


cnd_corr = {}
cnd_nvc= {}
cnd_df["task_desc"] = cnd_df[["task", "counterfactual"]].apply(get_cond_desc, axis=1)
cnd_df["solution"] = cnd_df["task"].apply(get_cond_solution)
cnd_df["correct"] = cnd_df["response"] == cnd_df["solution"]

cnd_df["pos_resp"] = cnd_df["response"].apply(lambda x: x == "positive")
cnd_df["neg_resp"] = cnd_df["response"].apply(lambda x: x == "negative")
cnd_df["nvc_resp"] = cnd_df["response"].apply(lambda x: x == "nvc")

for vp, df in cnd_df.groupby("id"):
    if vp not in shared_vp:
        continue
    cnd_corr[vp] = [df["correct"].mean()]
    cnd_nvc[vp] = [df["nvc_resp"].mean()]

# -------------------------------------------------------------------------
# Spatial
# -------------------------------------------------------------------------
spat_corr = {}
spat_indet = {}
spat_continuous = {}
spat_valid = {}
spat_nvc = {}
spatial_memory_corr = {}
spatial_shown_corr = {}
spatial_df["nvc_resp"] = spatial_df["response"].apply(lambda x: x == "nvc")
spatial_memorize_df = spatial_df[spatial_df["memorize_prem"]]
spatial_nomemorize_df = spatial_df[~spatial_df["memorize_prem"]]

for vp, df in spatial_memorize_df.groupby("id"):
    if vp not in shared_vp:
        continue
    spatial_memory_corr[vp] = [df["correct"].mean()]

for vp, df in spatial_nomemorize_df.groupby("id"):
    if vp not in shared_vp:
        continue
    spatial_shown_corr[vp] = [df["correct"].mean()]

for vp, df in spatial_df.groupby("id"):
    if vp not in shared_vp:
        continue
    spat_corr[vp] = [df["correct"].mean()]
    spat_indet[vp] = [df[df["indeterminate"]]["correct"].mean()]
    spat_valid[vp] = [df[df["valid"]]["correct"].mean()]
    spat_continuous[vp] = [df[df["continuous"]]["correct"].mean()]
    spat_nvc[vp] = [df["nvc_resp"].mean()]

# -------------------------------------------------------------------------
# Carddir
# -------------------------------------------------------------------------
carddir_df["is_nvc"] = carddir_df["response"].apply(lambda x: x == "nvc")
carddir_nvc = {}
for vp, df in carddir_df.groupby("id"):
    if vp not in shared_vp:
        continue
    carddir_nvc[vp] = [df["is_nvc"].mean()]
