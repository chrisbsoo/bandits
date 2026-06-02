import numpy as np

class EXP3_policy:
    def __init__(self, model, gamma=0.1):
        self.model = model
        self.gamma = gamma
        self.weights = {}

    def _initialize_new_arms(self, active_arms):
        for arm in active_arms:
            if arm not in self.weights:
                self.weights[arm] = 1.0

    def select_arm(self):
        active_arms = list(self.model.active_dat.keys())
        if not active_arms:
            raise ValueError("No active arms available to select.")
            
        self._initialize_new_arms(active_arms)
        
        active_weights = np.array([self.weights[arm] for arm in active_arms])
        sum_active_weights = np.sum(active_weights)
        
        # exp3 Formula: (1 - gamma) * (weight / sum_weights) + (gamma / K_active)
        K_active = len(active_arms)
        probabilities = (1.0 - self.gamma) * (active_weights / sum_active_weights) + (self.gamma / K_active)
        
        probabilities /= np.sum(probabilities)
        chosen_arm = np.random.choice(active_arms, p=probabilities)
        chosen_prob = probabilities[active_arms.index(chosen_arm)]
        return chosen_arm, chosen_prob

    def step(self, time):
        chosen_arm, prob = self.select_arm()
        reward = self.model.play(chosen_arm, time)
        estimated_reward = reward / prob
        K_active = len(self.model.active_dat)
        self.weights[chosen_arm] *= np.exp((self.gamma * estimated_reward) / K_active)
        
        if self.weights[chosen_arm] > 1e100:
            self._normalize_weights()
            
        return str(chosen_arm), reward

    def _normalize_weights(self):
        max_w = max(self.weights.values())
        for arm in self.weights:
            self.weights[arm] /= max_w