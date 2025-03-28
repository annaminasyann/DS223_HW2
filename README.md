# A/B Testing with Multi-Armed Bandits

This project simulates an A/B test using two algorithms:

- **Epsilon-Greedy:** Chooses a random arm sometimes (with chance 1/t) and otherwise picks the best arm so far.
- **Thompson Sampling:** Uses a probability model to choose an arm based on past rewards.

The code runs 20,000 trials for four bandit arms with true rewards of 1, 2, 3, and 4.

## What the Project Does

- **Learning Process:** Shows how the estimated rewards for each arm change over time.
- **Performance Comparison:** Compares the total rewards and regrets (how much reward is lost) of the two methods.
- **CSV Files:** Saves the trial data into three CSV files:
  - `bandit_rewards_EpsilonGreedy.csv`
  - `bandit_rewards_ThompsonSampling.csv`
  - `bandit_all_rewards.csv` (combined data)

## Requirements

The project requires the following packages:
- numpy
- pandas
- matplotlib
- loguru (version 0.7.2)

These dependencies are listed in the requirements.txt file.


## How to Run

- Clone this repo.
- Install requirements.
- Run Bandit.py.


## Author

Anna Minasyan
