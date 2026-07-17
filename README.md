# Data-Driven Construction of Individualized Process Models for Human Reasoning

This repository contains the data and analysis for the 2026 paper "Data-Driven Construction of Individualized Process Models for Human Reasoning".

## Getting Started
The analysis needs Python 3.10 or above which can be downloaded from the [official webpage](https://www.python.org/downloads/). 
Then, you may want to create a virtual environment, or you can install the dependencies directly.
The dependencies can be installed by running:

```
pip install -r requirements.txt
```

This also installs [CCOBRA](https://github.com/CognitiveComputationLab/ccobra), which we use to perform our analysis.

## Structure

- `Analysis`: Contains all scripts and results needed to replicate the analysis.
    - `Benchmark`: Contains the benchmark specification and models to be run with CCOBRA.
        - `benchmark.json`: Specification of the benchmark for CCOBRA. Can be run with CCOBRA directly.
    - `ProcessModel`: Contains the Process Model Generation method presented in the paper. The Usage is described in detail in a section below.
        - `actions`: Contains the implementations of the actions, as well as the base classes.
        - `apg_model.py`: Implementation of the global model generation method.
        - `subprocesses_planner.py`: Implementation of the local model search method.
    - `Results`: Contains the results of our global model evaluation
    - `solvability`: Contains the logfiles/traces of the local model evaluation (i.e., how many of the observations can be accounted for by searching a sequence of actions).
- `Data`: Contains the dataset.
    - `2025_crossdomain`: Contains the full cross domain dataset from previous work (Brand & Ragni, 2025).
    - `2025_syllog_mc.csv`: Contains the syllogistic dataset used in the paper.
    - `domain_info.py`: Helper file for preprocessing.
    - `preprocess_data.py`: Generates `2025_syllog_mc.csv` by bringing the respective original data in the format required by CCOBRA (which the global model evaluation is based on).
- `Paper`: Contains the paper as PDF.
- `Poster`: Contains the poster as PDF.

## Replicating the local model analysis

Navigate to the `Analysis` subfolder and run:
```
python solvability.py ../Data/2025_syllog_mc.csv
```
Respective traces will be placed in the solvability subfolder. The script searches for each observation sequences of actions that generate them from the task.

In order to see how a specific observation for a given task can be constructed, a search/plan can be done via a CLI-tool. Run for example:

```
python plan_cli.py "AA1" "Aac, Aca"
```

in order to find a sequence of actions to generate *Aac* and *Aca* from the syllogism *AA1*.

## Replicating the global model evaluation

Navigate to the `Benchmark` subfolder and run CCOBRA:
```
ccobra benchmark.json
```

An HTML-file is generated (similar to the one in the `Results`-folder), which allows to download the parameters (as JSON) and the scores (as CSV). Both files from the run used in the paper are included in the `Results`-folder.

**Note:** Depending on the way you installed CCOBRA, the executable may not have been added to your PATH variables, so it is not available directly. In those cases, you can either call the respective executable, or run the script `run_with_ccobra.py` directly. It works by spoofing the arguments and running CCOBRA directly on the benchmark. Simply run without any arguments:
```
python run_with_ccobra.py
```
Please consider that it is normal that the same participant is finished multiple times: Due to the leave-one-out cross-validation, CCOBRA runs the evaluation on the same participant once per task, which means 64 runs in our case.

### Importances of the actions

To generate the percentages used in Figure 1 in the paper, run the script `action_importance.py`. First, navigate to the `Analysis` folder, then run:
```
python action_importance.py Results/pipeline_fit.json
```
to print the percentages of our analysis (or use a different JSON file from another run instead).

## Creating the performance plot
Navigate to the `Analysis` subfolder and run:
```
python plot_performance.py [results.csv]
```
The CSV-file is optional, if not provided, the results of our paper will be loaded (`Results/results_loocv.csv`).

## References

- Brand, D., Bombach, C., & Ragni, M. (2026). Data-Driven Construction of Individualized Process Models for Human Reasoning. 

- Brand, D., & Ragni, M. (2025). Using Cross-Domain Data to Predict Syllogistic Reasoning Behavior. *Proceedings of the 47th Annual Meeting of the Cognitive Science Society*.