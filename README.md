# pyreason-rl-sim

PyReason-Gym (Gym environment wrapper for Pyreason: Symbolic Logic Enginer) based experiment scripts to train and evaluate RL agents. Training is based on a version of Deep-Q-Network algorithm.

## Getting Started

Make sure `python>=3.9.12` has been installed using the instructions found [here](https://www.python.org/downloads/release/python-3100/)

And to make it easier to install all the dependencies please use `pip>=22.2.2` using the instructions found [here](https://pip.pypa.io/en/stable/installation/)

Install all the dependencies using:

```bash
pip install -r requirements.txt
```

After which install Pyreason Gym using:

### PyReason Gym

Use the `main` branch of pyreason-gym

```bash
git clone https://github.com/lab-v2/pyreason-gym
cd pyreason-gym
cd ..
pip install -e pyreason-gym
```

## Basic Repository Structure

The folders `Training_scripts` and `Eval_scripts` house the 2 sets of scripts, for training RL agent and evaluating learned policies respectively. Each containing 4 files for the following kind of experiment:

1. Movement only, Markovian dynamics - file suffix:`_move_markov_`
2. Movement only, Non-Markovian dynamics - file suffix:`_move_non_markov_`
3. Movement + Shooting, Markovian dynamics - file suffix:`_shoot_markov_`
4. Movement + Shooting, Non-Markovian dynamics - file suffix:`_shoot_non_markov_`

## Usage

### Execution steps to run experiments

1. Run one of the training scripts using: `python <script_name>.py`. This creates policy files for corresponding steps with `.pth` extension.
2. Optional: Move these `.pth` files to a subfolder and then point to this folder in evaluation script.
3. Run the corresponding evaluation script with same suffix as training script file. This generates a `.json` file with all the evaluation metrics.
4. Change the input file name in `win_plotter.py` script to the newly created `.json` file and run the script to generate a plot of the avg. scores and win percentages for various poilicies. The output will be save in a `.jpg` file.
