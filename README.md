# Boosted Control Functions: <br> Distribution generalization and invariance in confounded models


The goal of this repository is to reproduce the numerical experiments from <a href="#ref1">[1]</a><a id="ref1-back"></a>.

If you use this software in your work, please cite it using the following metadata.

```bibtex
@misc{gnecco2024boostedcontrolfunctionsdistribution,
      title={Boosted Control Functions: Distribution generalization and invariance in confounded models}, 
      author={Nicola Gnecco and Jonas Peters and Sebastian Engelke and Niklas Pfister},
      year={2024},
      eprint={2310.05805},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2310.05805}, 
}
```

## Table of Contents
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)
- [Instructions](#instructions)
    - [Experiment 1: Predicting unseen interventions](#experiment-1-predicting-unseen-interventions)
    - [Experiment 2: Estimating M0](#experiment-2-estimating-m0)
    - [California housing dataset](#california-housing-dataset)
- [Contributors](#contributors)
- [License](#license)
- [References](#references)

## Directory Structure
```
.
├── R-code     
│   ├── R               # R functions
│   └── main            # R scripts
|
├── data
│   ├── processed       # Processed data
│   └── raw             
│       └── prism_temp  # Temperature data 
|
├── python
│   ├── main            # Python scripts
│   └── src             
│       ├── bcf         # Classes and functions for BCF
│       ├── scenarios   # Functions to generate data
│       ├── simulations # Functions to run simulations
│       └── utils       # Helper functions
|
└── results
    ├── figures         # Saved figures
    └── output_data     # Saved results
```


## Dependencies

- Python: version (3.9.1)
- R: version (4.0.2)

### Installing Python Requirements
1. **Ensure Python and pip are installed:**
Before installing the requirements, make sure you have Python installed on your system. You can check by running in the terminal:
```bash
python --version
```
Also, ensure you have **`pip`** (Python package manager) installed by running in the terminal:
```bash
pip --version
```
If you don't have Python or pip installed, please [download Python](https://www.python.org/downloads/) and install it. Pip is included in Python versions 3.4+.

2. **Navigate to python directory:**
Use the terminal to navigate to the directory containing your [`requirements.txt`](./python/requirements.txt) file.
```bash
cd python
```

3. **Install the requirements:**
Once inside the directory, run the following command to install the required packages:
```bash
pip install -r requirements.txt

pip install -e . 
```

The first line will install all the Python packages listed in [`requirements.txt`](./python/requirements.txt). 
The second line will install the python code located in [`./python/src/`](./python/src/) as a package.

:warning: Consider [creating a new Python environment](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) before installing the Python packages.

### Installing R Requirements
1. **Ensure R is installed:**
Make sure you have R installed on your system by running in the terminal:
```bash
R --version
```
If you don't have R installed please [download R](https://cran.rstudio.com/) and install it.

2. **Install the requirements:**
Move to the R-code directory and launch the R script [`dependencies.R`](./R-code/main/dependencies.R) by running in the terminal:
```bash
cd R-code
Rscript --vanilla main/dependencies.R
```



## Instructions

:warning: Experiment 1 and 2 were originally run on a cluster with a [PBS/TORQUE](http://docs.adaptivecomputing.com/torque/6-0-4/adminGuide/torquehelp.htm#topics/torque/0-intro/introduction.htm) job schedular. The instructions below only apply to this type of scheduler.


### Experiment 1: Predicting unseen interventions

:warning: This experiment was run in parallel using 80 cores and took around 2 hours.

**1a) How to run on a PBS/TORQUE cluster:**

- Open [`./python/main/run_job.sh`](./python/main/run_job.sh) and modify lines 29–31 depending on your cluster.
- From the root of the project, in the terminal type the following:
```bash
cd python

qsub main/run_job.sh "python main/experiment_1.py"
```

**1b) How to run locally**

- Open  [`./python/main/experiment_1.py`](./python/main/experiment_1.py) and on line 13 set the number of cores according to your machine, e.g., `N_WORKERS = 4`.

- From the root of the project, in the terminal type the following:
```bash
cd python

python main/experiment_1.py
```

:bulb: To run only a few iterations of the experiment, change the number of repetitions on line 29 of [`./python/main/experiment_1.py`](./python/main/experiment_1.py), e.g.,  `"n_reps": range(2)`.


**2) Plot results**
- From the root of the project, in the terminal type the following:
```bash
cd R-code

Rscript --vanilla main/plot-interv_sim.R
```



### Experiment 2: Estimating M0

:warning: This experiment was run in parallel using 10 cores and took around 20 minutes.

**1a) How to run on a PBS/TORQUE cluster:**

- Open [`./python/main/run_job.sh`](./python/main/run_job.sh) and modify lines 29–31 depending on your cluster.
- From the root of the project, in the terminal type the following:
```bash
cd python

qsub main/run_job.sh "python main/experiment_2.py"
```

**1b) How to run locally**

- Open [`./python/main/experiment_2.py`](./python/main/experiment_2.py) and on line 13 set the number of cores according to your machine, e.g., `N_WORKERS = 4`.

- From the root of the project, in the terminal type the following:
```bash
cd python

python main/experiment_2.py
```

:bulb: To run only a few iterations of the experiment, change the number of repetitions on line 31 of [`./python/main/experiment_2.py`](./python/main/experiment_2.py), e.g.,  `"n_reps": range(2)`.


**2) Plot results**
- From the root of the project, in the terminal type the following:
```bash
cd R-code

Rscript --vanilla main/plot-nullspace_sim.R
```



### California housing dataset

To run the experiment and produce the figures for the California housing dataset, follow these steps. 

```bash
cd python

python main/housing_data-import.py

python main/asc2csv.py ../data/raw/prism_temp/ "../data/processed/temp-data-california.csv" "Mean_Temp"

cd ../R-code

Rscript --vanilla main/prepare-temp_housing_data.R

cd ../python

python main/housing_data_analysis_regions.py

cd ../R-code

Rscript --vanilla main/plot-temp_housing_data.R
```

## Contributors

- [Nicola Gnecco](https://ngnecco.com)
- [Jonas Peters](https://people.math.ethz.ch/~jopeters/)
- [Sebastian Engelke](http://www.sengelke.com/)
- [Niklas Pfister](https://niklaspfister.github.io/)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.


## References
<a id="ref1"></a><a href="#ref1-back">[1]</a>: Nicola Gnecco, Jonas Peters, Sebastian Engelke, and Niklas Pfister. 2024. "Boosted Control Functions: Distribution generalization and invariance in confounded models." arXiv Preprint [https://arxiv.org/abs/2310.05805].

