from abc import ABC, abstractmethod
import numpy as np
from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt


class Bandit(ABC):
    """
    Abstract base class for bandit algorithms.

    Defines the interface for bandit algorithms that interact with multiple arms.
    Each subclass must implement methods to select an arm, update estimates, run experiments, and report results.
    """

    @abstractmethod
    def __init__(self, p):
        """
        Initialize the bandit algorithm with true reward means for each arm.

        :param p: List or array of true mean rewards for each bandit arm.
        :type p: list[float] or numpy.ndarray
        """
        pass

    @abstractmethod
    def __repr__(self):
        """
        Provide a string representation of the algorithm (for logging and identification).

        :return: Name of the algorithm.
        :rtype: str
        """
        pass

    @abstractmethod
    def pull(self):
        """
        Select an arm to pull according to the algorithm's strategy and simulate receiving a reward.

        :return: A tuple containing the index of the chosen arm and the reward obtained from that arm.
        :rtype: tuple(int, float)
        """
        pass

    @abstractmethod
    def update(self):
        """
        Update the algorithm's internal estimates based on the most recently obtained reward.

        This typically updates counts and reward estimates for the arm that was last pulled.
        """
        pass

    @abstractmethod
    def experiment(self, num_trials):
        """
        Run the bandit algorithm for a given number of trials.

        This repeatedly selects arms, observes rewards, and updates estimates, while recording the history.

        :param num_trials: The number of arm pulls (trials) to execute.
        :type num_trials: int
        """
        pass

    @abstractmethod
    def report(self):
        """
        Summarize the experiment results by logging key metrics and saving trial data to a CSV file.

        The report typically includes cumulative reward, cumulative regret, and possibly average reward/regret.
        It also saves the detailed history of trials (arm choices and rewards) to a CSV for further analysis.
        """
        pass


class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy algorithm with decaying exploration rate (epsilon = 1/t).

    Selects a random arm with probability epsilon (exploration) or the current best estimated arm with probability 1 - epsilon (exploitation).
    Epsilon decays over trials, encouraging more exploitation over time.
    """

    def __init__(self, p):
        """
        Initialize the Epsilon-Greedy algorithm.

        :param p: True mean rewards for each arm.
        :type p: list[float] or numpy.ndarray
        """
        self.true_rewards = np.array(p, dtype=float)
        self.n_arms = len(self.true_rewards)
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.estimates = np.zeros(self.n_arms, dtype=float)
        self.history = []             # List of dicts for each trial: {"Trial", "Bandit", "Reward", "Algorithm"}.
        self.history_estimates = []   # Snapshots of estimates after each trial.
        self.cum_rewards = []         # Cumulative reward after each trial.
        self.total_reward = 0.0
        self.trial = 1                # Start trial counter at 1.
        logger.debug("Initialized EpsilonGreedy with {} arms.".format(self.n_arms))

    def __repr__(self):
        """
        Return a string representation of the Epsilon-Greedy algorithm.

        :return: Identifier string for this algorithm.
        :rtype: str
        """
        return "EpsilonGreedy Algorithm"

    def pull(self):
        """
        Decide which arm to pull using decaying epsilon = 1/trial.

        If exploring (with probability epsilon), a random arm is selected;
        if exploiting, the arm with the highest current estimate is chosen.

        :return: The index of the chosen arm and the simulated reward.
        :rtype: tuple(int, float)
        """
        epsilon = 1.0 / float(self.trial)
        if np.random.rand() < epsilon:
            chosen_arm = np.random.randint(0, self.n_arms)
            logger.debug(f"Trial {self.trial}: Exploring – chose arm {chosen_arm} (ε={epsilon:.3f}).")
        else:
            chosen_arm = int(np.argmax(self.estimates))
            logger.debug(f"Trial {self.trial}: Exploiting – chose arm {chosen_arm} (est={self.estimates[chosen_arm]:.3f}).")
        reward = np.random.normal(loc=self.true_rewards[chosen_arm], scale=1.0)
        self.last_arm = chosen_arm
        self.last_reward = reward
        return chosen_arm, reward

    def update(self):
        """
        Update the estimated reward for the last pulled arm using an incremental average.
        """
        arm = getattr(self, 'last_arm', None)
        reward = getattr(self, 'last_reward', None)
        if arm is None or reward is None:
            return
        self.counts[arm] += 1
        n = self.counts[arm]
        self.estimates[arm] += (reward - self.estimates[arm]) / float(n)
        logger.debug(f"Trial {self.trial}: Updated estimate for arm {arm} = {self.estimates[arm]:.3f}.")

    def experiment(self, num_trials=20000):
        """
        Execute the Epsilon-Greedy algorithm for a specified number of trials.
        """
        logger.info(f"Starting experiment for {self} with {num_trials} trials.")
        for _ in range(num_trials):
            chosen_arm, reward = self.pull()
            self.update()
            self.total_reward += reward
            self.history.append({
                "Trial": self.trial,
                "Bandit": chosen_arm,
                "Reward": reward,
                "Algorithm": repr(self)
            })
            self.cum_rewards.append(self.total_reward)
            self.history_estimates.append(self.estimates.copy())
            self.trial += 1
        logger.info(f"Experiment completed for {self}.")

    def report(self):
        """
        Log experiment summary statistics and save trial history to a CSV file.
        """
        num_trials = self.trial - 1
        best_mean = np.max(self.true_rewards)
        per_trial_regret = [max(0.0, best_mean - record["Reward"]) for record in self.history]
        cum_regret = np.cumsum(per_trial_regret)
        total_regret = cum_regret[-1] if len(cum_regret) > 0 else 0.0
        avg_reward = self.total_reward / num_trials if num_trials > 0 else 0.0
        avg_regret = total_regret / num_trials if num_trials > 0 else 0.0
        df = pd.DataFrame(self.history)
        df.to_csv("bandit_rewards_EpsilonGreedy.csv", index=False)
        logger.info("Data stored in bandit_rewards_EpsilonGreedy.csv.")
        logger.info(f"{repr(self)}: Final Cumulative Reward = {self.total_reward:.2f}")
        logger.info(f"{repr(self)}: Final Cumulative Regret = {total_regret:.2f}")
        logger.info(f"{repr(self)}: Average Reward = {avg_reward:.2f}")
        logger.info(f"{repr(self)}: Average Regret = {avg_regret:.2f}")


class ThompsonSampling(Bandit):
    """
    Thompson Sampling algorithm with Gaussian prior and known precision.

    Assumes rewards are drawn from a Gaussian distribution with variance 1.
    Uses a conjugate Gaussian prior for each arm's mean.
    """

    def __init__(self, p, prior_mean=0.0, prior_precision=1.0):
        """
        Initialize the Thompson Sampling algorithm.

        :param p: True mean rewards for each arm.
        :type p: list[float] or numpy.ndarray
        :param prior_mean: Prior mean for each arm.
        :type prior_mean: float, optional
        :param prior_precision: Prior precision (1/variance) for each arm.
        :type prior_precision: float, optional
        """
        self.true_rewards = np.array(p, dtype=float)
        self.n_arms = len(self.true_rewards)
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.sum_rewards = np.zeros(self.n_arms, dtype=float)
        self.prior_mean = prior_mean
        self.prior_precision = prior_precision
        self.history = []
        self.history_estimates = []  # Posterior means for each arm per trial.
        self.cum_rewards = []
        self.total_reward = 0.0
        self.trial = 1
        logger.debug("Initialized ThompsonSampling with {} arms (prior_mean={}, prior_precision={})."
                     .format(self.n_arms, self.prior_mean, self.prior_precision))

    def __repr__(self):
        return "ThompsonSampling Algorithm"

    def pull(self):
        """
        Select an arm by sampling from each arm's posterior and choosing the highest.
        """
        samples = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            posterior_precision = self.prior_precision + self.counts[i]
            posterior_mean = (self.prior_precision * self.prior_mean + self.sum_rewards[i]) / posterior_precision
            posterior_std = np.sqrt(1.0 / posterior_precision)
            samples[i] = np.random.normal(posterior_mean, posterior_std)
            logger.debug(f"Trial {self.trial}: Arm {i} posterior μ={posterior_mean:.3f}, σ={posterior_std:.3f}, sample={samples[i]:.3f}")
        chosen_arm = int(np.argmax(samples))
        logger.debug(f"Trial {self.trial}: ThompsonSampling chose arm {chosen_arm} (sample={samples[chosen_arm]:.3f})")
        reward = np.random.normal(loc=self.true_rewards[chosen_arm], scale=1.0)
        self.last_arm = chosen_arm
        self.last_reward = reward
        return chosen_arm, reward

    def update(self):
        """
        Update the posterior parameters for the arm that was last pulled.
        """
        arm = getattr(self, 'last_arm', None)
        reward = getattr(self, 'last_reward', None)
        if arm is None or reward is None:
            return
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward
        logger.debug(f"Trial {self.trial}: Updated arm {arm} -> count={self.counts[arm]}, sum_rewards={self.sum_rewards[arm]:.3f}")

    def experiment(self, num_trials=20000):
        """
        Execute the Thompson Sampling algorithm for a specified number of trials.
        """
        logger.info(f"Starting experiment for {self} with {num_trials} trials.")
        for _ in range(num_trials):
            chosen_arm, reward = self.pull()
            self.update()
            self.total_reward += reward
            self.history.append({
                "Trial": self.trial,
                "Bandit": chosen_arm,
                "Reward": reward,
                "Algorithm": repr(self)
            })
            self.cum_rewards.append(self.total_reward)
            current_post_means = np.zeros(self.n_arms)
            for i in range(self.n_arms):
                posterior_precision = self.prior_precision + self.counts[i]
                current_post_means[i] = (self.prior_precision * self.prior_mean + self.sum_rewards[i]) / posterior_precision
            self.history_estimates.append(current_post_means)
            self.trial += 1
        logger.info(f"Experiment completed for {self}.")

    def report(self):
        """
        Log experiment summary statistics and save trial history to a CSV file.
        """
        num_trials = self.trial - 1
        best_mean = np.max(self.true_rewards)
        per_trial_regret = [max(0.0, best_mean - record["Reward"]) for record in self.history]
        cum_regret = np.cumsum(per_trial_regret)
        total_regret = cum_regret[-1] if len(cum_regret) > 0 else 0.0
        avg_reward = self.total_reward / num_trials if num_trials > 0 else 0.0
        avg_regret = total_regret / num_trials if num_trials > 0 else 0.0
        df = pd.DataFrame(self.history)
        df.to_csv("bandit_rewards_ThompsonSampling.csv", index=False)
        logger.info("Data stored in bandit_rewards_ThompsonSampling.csv.")
        logger.info(f"{repr(self)}: Final Cumulative Reward = {self.total_reward:.2f}")
        logger.info(f"{repr(self)}: Final Cumulative Regret = {total_regret:.2f}")
        logger.info(f"{repr(self)}: Average Reward = {avg_reward:.2f}")
        logger.info(f"{repr(self)}: Average Regret = {avg_regret:.2f}")


class Visualization:
    """
    Visualization utility class for plotting the results of bandit experiments.

    Provides methods to plot the learning curves (estimated values per arm) and the
    cumulative rewards/regrets of two bandit algorithms for comparison.
    """

    def plot1(self, history_estimates, algorithm_name, true_rewards):
        """
        Plot the learning process (estimated values) for each arm over time using a linear scale.

        :param history_estimates: Sequence of estimate vectors (one per trial) for the algorithm.
        :type history_estimates: list or numpy.ndarray
        :param algorithm_name: Name of the algorithm.
        :type algorithm_name: str
        :param true_rewards: True mean rewards for each arm.
        :type true_rewards: list[float] or numpy.ndarray
        """
        history_estimates = np.array(history_estimates)
        trials = history_estimates.shape[0]
        t_axis = np.arange(1, trials + 1)

        plt.figure(figsize=(10, 6))
        for i in range(history_estimates.shape[1]):
            plt.plot(t_axis, history_estimates[:, i], label=f'Arm {i} (true {true_rewards[i]})')
        plt.xlabel("Trial")
        plt.ylabel("Estimated Mean Reward")
        plt.title(f"Learning Curve ({algorithm_name})")
        plt.legend()
        plt.grid(True)
        plt.show()
        logger.info(f"Displayed learning curve for {algorithm_name}.")

    def plot2(self, cum_rewards_eg, cum_rewards_ts, cum_regret_eg, cum_regret_ts):
        """
        Plot cumulative rewards and cumulative regrets for Epsilon-Greedy vs Thompson Sampling.

        Produces two plots: one for cumulative rewards and one for cumulative regrets.

        :param cum_rewards_eg: Cumulative rewards per trial for Epsilon-Greedy.
        :type cum_rewards_eg: list or numpy.ndarray
        :param cum_rewards_ts: Cumulative rewards per trial for Thompson Sampling.
        :type cum_rewards_ts: list or numpy.ndarray
        :param cum_regret_eg: Cumulative regrets per trial for Epsilon-Greedy.
        :type cum_regret_eg: numpy.ndarray
        :param cum_regret_ts: Cumulative regrets per trial for Thompson Sampling.
        :type cum_regret_ts: numpy.ndarray
        """
        trials = len(cum_rewards_eg)
        t_axis = np.arange(1, trials + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(t_axis, cum_rewards_eg, label="Epsilon-Greedy")
        plt.plot(t_axis, cum_rewards_ts, label="Thompson Sampling")
        plt.xlabel("Trial")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Reward Comparison")
        plt.legend()
        plt.grid(True)
        plt.show()
        logger.info("Displayed cumulative rewards plot.")

        plt.figure(figsize=(8, 6))
        plt.plot(t_axis, cum_regret_eg, label="Epsilon-Greedy")
        plt.plot(t_axis, cum_regret_ts, label="Thompson Sampling")
        plt.xlabel("Trial")
        plt.ylabel("Cumulative Regret")
        plt.title("Cumulative Regret Comparison")
        plt.legend()
        plt.grid(True)
        plt.show()
        logger.info("Displayed cumulative regrets plot.")


def comparison():
    """
    Run both Epsilon-Greedy and Thompson Sampling algorithms, compare their performance, and visualize results.

    This function sets up the bandit problem, runs experiments for both algorithms,
    logs summary results (and saves CSV files), and generates learning curve and cumulative performance plots.
    """
    bandit_means = [1, 2, 3, 4]
    num_trials = 20000

    eg = EpsilonGreedy(bandit_means)
    ts = ThompsonSampling(bandit_means, prior_mean=0.0, prior_precision=1.0)

    eg.experiment(num_trials=num_trials)
    ts.experiment(num_trials=num_trials)

    eg.report()
    ts.report()

    all_history = pd.DataFrame(eg.history + ts.history)
    all_history = all_history[["Bandit", "Reward", "Algorithm"]]
    all_history.to_csv("bandit_all_rewards.csv", index=False)
    logger.info(f"Combined data saved to bandit_all_rewards.csv (total {len(all_history)} rows).")

    viz = Visualization()
    viz.plot1(eg.history_estimates, "Epsilon-Greedy", bandit_means)
    viz.plot1(ts.history_estimates, "Thompson Sampling", bandit_means)

    best_mean = max(bandit_means)
    cum_regret_eg = np.cumsum([max(0.0, best_mean - rec["Reward"]) for rec in eg.history])
    cum_regret_ts = np.cumsum([max(0.0, best_mean - rec["Reward"]) for rec in ts.history])

    viz.plot2(eg.cum_rewards, ts.cum_rewards, cum_regret_eg, cum_regret_ts)


if __name__ == "__main__":
    comparison()
