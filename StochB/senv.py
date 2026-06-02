import numpy as np

# INPUT: "k1p1 k2p1 k3p1 ... ", "k1p2 k2p2 k3p2 ..."

class stochastic_model:
    def __init__(self, dist, seed, *params):
        np.seed = seed
        # Initializes arms with parameters
        self.dists = {"bern": [np.random.binomial, 1], "dnorm": [np.random.normal, 2], 
                    "exp": [np.random.exponential, 1], "beta" : [np.random.beta, 2]}

        self.data = {}
        if dist in self.dists:
            x = self.dists[dist]
            self.dist = x[0]
            self.dim = x[1]

        if len(params) > 0:
            self.add(*params)
        
    def add(self, *params):
        # Adds a new arm with parameters
        plist = [p.split(" ") for p in params]
        plist = np.array(plist).T

        for i, x in enumerate(plist):
            idx = str(len(self.data))
            self.data[idx] = [float(k) if "." in k else int(k) for k in x]
        
        self.active_dat = self.data.copy()

    def active(self, idxs):
        # Set Active arms
        ilist = idxs.split(" ")
        self.active_dat = {x : self.data[x] for x in ilist}

    def remove(self, idx):
        # Removes an arm given index
        del self.data[str(idx)]

        self.active_dat = self.data.copy()
    
    def play(self, idx):
        # Plays an arm given index
        params = self.active_dat[str(idx)]
        if len(params) == self.dim:
            res = self.dist(*params)
            return res
        else:
            return "parameter sizes are different"
        
    def play_multi(self, idx, n):
        # Plays multiple arms given index
        for _ in range(n): self.play(idx)


