import numpy as np

class EXP3_policy:
    def __init__(self, model, rng, gamma=None):
        self.K = model.arm_size
        self.T = model.horizon
        self.model = model
        self.weights = {}
        self.rng = rng

        if gamma is None:
            numer = self.K * np.log(self.K)
            denom = (np.e - 1) * self.T

            threshold = np.sqrt(numer / denom)
            gamma = min(1, threshold)

        self.gamma = gamma
    
    def init_weights(self):
        for i in range(self.K):
            if str(i) not in self.weights:
                self.weights[str(i)] = [1]
    
    def update_weights(self, chosen, est, time):
        for arm in self.weights:
            if arm == chosen:
                w = self.weights[chosen][time]
                exp_term = (self.gamma * est) / self.K
                self.weights[chosen].append(w * np.exp(exp_term))
            else:
                self.weights[arm].append(self.weights[arm][time])
    
    def step(self, time):
        self.init_weights()
        probs = {}

        weight_sum = 0
        for i in range(self.K):
            weight_sum += self.weights[str(i)][time]

        for i in range(self.K):
            term1 = ((1-self.gamma) * self.weights[str(i)][time]) / weight_sum
            term2 = self.gamma / self.K
            probs[str(i)] = term1 + term2
        
        arm_idxs = list(probs)
        prob_vals = list(probs.values())

        prob_vals = prob_vals / sum(prob_vals) # renormalisation
        
        chosen = self.rng.choice(arm_idxs, p=prob_vals)
        reward = self.model.play(chosen, time)

        weight_vals = list(self.weights.values())

        # check if time is len, if it is, we update the next weight
        # note this does not mean same time gives same result, because of algo's internal RNG

        if len(list(weight_vals[0])) - 1 == time:
            reward_est = reward / probs[chosen]
            self.update_weights(chosen, reward_est, time)
        
        return chosen, reward
    





    


