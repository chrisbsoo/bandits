import numpy as np

# REMEMBER FOR MODELS: MUST HAVES:
# 1. TIME HORIZON
# 2. NUMBER OF ARMS, INDEX OF ARMS (0 ... K-1)
# 3. STEP FUNCTION, RETURNS REWARD

class stochastic_model:
    def __init__(self, dist, rng, horizon, *params):
        # Initializes arms with parameters
        self.dists = {"bin": [rng.binomial, 2], "dnorm": [rng.normal, 2], 
                    "exp": [rng.exponential, 1], "beta" : [rng.beta, 2]}
        
        self.rng = rng
        self.horizon = horizon

        self.data = {}
        self.points = {}

        if dist in self.dists:
            x = self.dists[dist]
            self.dist = x[0]
            self.dim = x[1]

        if len(params) > 0:
            self.add_arm(*params)
        
    def add_arm(self, *params):
        # Adds a new arm with parameters
        plist = [p.split(" ") for p in params]
        plist = np.array(plist).T

        for i, x in enumerate(plist):
            idx = str(len(self.data))
            self.data[idx] = [float(k) if "." in k else int(k) for k in x]
        
        self.active_dat = self.data.copy()
    
    def generate(self):
        for k in list(self.data):
            self.points[k] = []

        for _ in range(self.horizon):
            for k in list(self.data):
                self.points[k].append(self.play_arm(k))

    def active(self, idxs):
        # Set Active arms
        ilist = idxs.split(" ")
        self.active_dat = {x : self.data[x] for x in ilist}

    def remove_arm(self, idx):
        # Removes an arm given index
        del self.data[str(idx)]

        self.active_dat = self.data.copy()
    
    def play_arm(self, idx):
        # Plays an arm given index
        params = self.active_dat[str(idx)]
        if len(params) == self.dim:
            res = self.dist(*params)
            return res
        else:
            raise ValueError("parameter sizes are different")
    
    def play(self, idx, time):
        # Plays an arm given index and time

        return self.points[idx][time]
    




