# Predict Customer Churn with Clean Code

## Description

This project is about Clean Code Principles. 
The problem is to predict credit card customers that are most likely to churn using clean code best practices.

## Prerequisites

Python and Jupyter Notebook are required.
Also a Linux environment may be needed within windows through WSL.

## Dependencies
- sklearn
- numpy
- pandas
- matplotlib
- seaborn
- pytest

## Installation

Use the package manager [conda](https://docs.conda.io/en/latest/) to install the dependencies from the ```conda.yml```

```bash
conda env create -f conda.yml
```

## Usage

The main script to run using the following command.
```bash
python churn_library.py
``` 
which will generate
- EDA plots in the directory ```./images/EDA/```
- Model metrics plots in the directory ```./images/results/```
- Saved model pickle files in the directory ```./models/```
- A log file ```./log/churn_library.log``` 

The tests script can be used with the following command which will generate a log file ```./churn_script_logging_and_tests.log``` 
```bash
python churn_script_logging_and_tests.py
```

## License
Distributed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt) License. See ```LICENSE``` for more information.
