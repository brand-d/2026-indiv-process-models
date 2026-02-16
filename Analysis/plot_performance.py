''' Script plotting the performance boxplots of the global model/leave-one-out CV run.
Usage to create the boxplot in the Paper (Figure 2):
python plot_performance.py
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

file = None
if len(sys.argv) > 1:
    file = sys.argv[1]
else:
    file = os.path.join("Results", "results_loocv.csv")

# Load the data
result_df = pd.read_csv(file)

result_df["hit"] = result_df["score_response"]

# Normalize model names
result_df['model'] = result_df['model'].replace({
    "PHM (indiv)": 'PHM',
    'mReasoner': 'mReasoner',
    'UniformModel': 'Random',
    "ProcessModel": "Gen. Process\nModel",
    "TransSet": "TransSet",
    "MFP": "Most freq.\nPattern"
})

med_df = result_df.groupby(['model', 'id'], as_index=False)['hit'].agg('mean')
print("Median")
print(med_df.groupby('model')['hit'].agg('median'))
print()
print("Mean")
print(med_df.groupby('model')['hit'].agg('mean'))
print()

# Mean the data
subj_df = result_df.groupby(
    ['model', 'id'], as_index=False)['hit'].agg('mean')

order_df = subj_df.groupby(['model'], as_index=False)['hit'].agg('mean')
order = order_df.sort_values('hit')['model']

# Prepare for plotting
sns.set_theme(style="whitegrid")
plt.figure(figsize=(6.5, 3.5))

# Plot the data
sns.swarmplot(x="model", y="hit", data=subj_df, order=order,
              dodge=True, linewidth=0.5, size=2, edgecolor=[0.3,0.3,0.3], color="midnightblue", zorder=1)

ax = sns.boxplot(x="model", y="hit", data=subj_df, order=order,
                 showcaps=True,boxprops={"facecolor": "midnightblue", "zorder":10, "alpha": 0.5, "edgecolor": "midnightblue"},
                 showfliers=False,whiskerprops={"zorder":10}, linewidth=1, color="black",
                 zorder=10, showmeans=True, meanprops={"markerfacecolor":"white", "markeredgecolor":"black"})

[label.set_fontweight('bold') for label in ax.get_xticklabels()]
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.xlabel("")
ax.set_ylabel("Jaccard Coefficient")
ax.tick_params(labelsize=12)

plt.tight_layout()
plt.savefig('loo_cv_plot.pdf')
plt.show()
