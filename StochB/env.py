import numpy as np

class bandit_model:
    def __init__(self, dist, *params):
        self.dists = {"bern": [np.random.binomial, 1], "dnorm": [np.random.normal, 2], 
                    "exp": [np.random.exponential, 1], "beta" : [np.random.beta, 2]}

        self.data = {}
        if dist in self.dists:
            x = self.dists[dist]
            self.dist = x[0]
            self.dim = x[1]

        print(f"Stochastic Bandit Initialised\nD: {self.dist}\n")

        if len(params) > 0:
            self.add(*params)
        
    def add(self, *params):
        plist = [p.split(" ") for p in params]
        plist = np.array(plist).T

        for i, x in enumerate(plist):
            idx = str(len(self.data))
            self.data[idx] = [float(k) if "." in k else int(k) for k in x]
    
        print(f"Stochastic Bandit\nK: {len(self.data)}\n")
        

    def remove(self, idx):
        del self.data[str(idx)]

        print(f"Stochastic Bandit\nK: {len(self.data)}\n")
    
    def play(self, idx):
        params = self.data[str(idx)]
        if len(params) == self.dim:
            res = self.dist(*params)
            print("Reward: ", res)
            return res
        else:
            return "parameter sizes are different"
