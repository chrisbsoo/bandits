import numpy as np

class UCB_policy:
    def __init__(self, model, c=2.0):
        self.model = model
        self.c = c

        self.total_counts = {}
        self.total_rewards = {}
        self.t = 0 

    def select_arm(self):
        active_arms = list(self.model.active_dat.keys())
        
        if not active_arms:
            raise ValueError("No active arms available to select.")

        for arm in active_arms:
            if arm not in self.total_counts:
                self.total_counts[arm] = 0
                self.total_rewards[arm] = 0.0

        for arm in active_arms:
            if self.total_counts[arm] == 0:
                return arm

        ucb_values = {}
        for arm in active_arms:
            mean_reward = self.total_rewards[arm] / self.total_counts[arm]
            bonus = np.sqrt((self.c * np.log(self.t)) / self.total_counts[arm])
            ucb_values[arm] = mean_reward + bonus

        return max(ucb_values, key=ucb_values.get)

    def step(self):
        chosen_arm = self.select_arm()
        
        reward = self.model.play(chosen_arm)
        
        self.t += 1
        self.total_counts[chosen_arm] += 1
        self.total_rewards[chosen_arm] += reward
        
        return chosen_arm, reward

    def run(self, iterations):
        cumm_rew = [0]
        arms = []
        for i in range(iterations):
            arm, reward = self.step()
            arms.append(arm)
            cumm_rew.append(round(cumm_rew[i] + reward, 2))

        return arms, cumm_rew