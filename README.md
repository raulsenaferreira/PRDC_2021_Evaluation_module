# Evaluation module
Evaluation/visualization module for generating graphs and results presented in the paper "Benchmarking Safety Monitors for Image Classifiers with Machine Learning"

## If you use this repository or part of it please consider to cite the respective paper
```
@article{ferreira2021benchmarking,
  title={Benchmarking Safety Monitors for Image Classifiers with Machine Learning},
  author={Ferreira, Raul Sena and Arlat, Jean and Guiochet, J{\'e}r{\'e}mie and Waeselynck, H{\'e}l{\`e}ne},
  journal={26th IEEE Pacific Rim International Symposium on Dependable Computing (PRDC 2021), Perth, Australia},
  year={2021}
}
```
## Simple installing
python -m venv env

.\env\Scripts\activate

pip install -r requirements.txt

P.S.: We use https://neptune.ai/ for storing and retrieving the results. Just change the username in the script neptune_config. However, it is quite easy to change the code to make it save in csv files instead of using a cloud tool.

## Configuration
config_id -> type=int, help="ID for a set of pre-defined visualizations"

path_for_saving_plots -> type=String, help="Root path for saving visualizations"

## Usage
python run_visualizations.py 1 plots
