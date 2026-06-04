import numpy as np

# INPUT: "k1p1 k2p1 k3p1 ... ", "k1p2 k2p2 k3p2 ..."

class stochastic_model:
    def __init__(self, dist, rng, horizon, *params):
        # Initializes arms with parameters
        self.dists = {"bern": [rng.random.binomial, 1], "dnorm": [rng.random.normal, 2], 
                    "exp": [rng.random.exponential, 1], "beta" : [rng.random.beta, 2]}
        
        self.rng = rng
        self.horizon = horizon

        self.data = {}
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
    
    @staticmethod
    def generate(self):
        

    def active(self, idxs):
        # Set Active arms
        ilist = idxs.split(" ")
        self.active_dat = {x : self.data[x] for x in ilist}

    def remove_arm(self, idx):
        # Removes an arm given index
        del self.data[str(idx)]

        self.active_dat = self.data.copy()
    
    def play(self, idx, time):
        # Plays an arm given index
        params = self.active_dat[str(idx)]
        if len(params) == self.dim:
            res = self.dist(*params)
            return res
        else:
            return "parameter sizes are different"



