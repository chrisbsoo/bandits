
import numpy as np

# INPUT: "k1t1 k1t2 k1t3 ... ", "k2t1 k2t2 k2t3 ..."
# 0-Based Indexxing

class adversial_model:
    def __init__(self, *rewards):
        self.data = {}

        for i, reward in enumerate(rewards):
            self.data[str(i)] = [int(x) for x in reward.split(" ")]
        
        self.horizon = len(self.data["0"])
        
        self.active_dat = self.data.copy()
        
    def add_arm(self, *rewards):
        for reward in rewards:
            idx = str(len(self.data))
            self.data[idx] = [int(x) for x in reward.split(" ")]
        
        self.active_dat = self.data.copy()
    
        print(f"Adversial Bandit\nK: {len(self.data)}\n")
    
    def active(self, idxs):
        # Set Active arms
        ilist = idxs.split(" ")
        self.active_dat = {x : self.data[x] for x in ilist}

    def remove_arm(self, idx):
        # Removes an arm given index
        del self.data[str(idx)]

        self.active_dat = self.data.copy()

        print(f"Adversial Bandit\nK: {len(self.data)}\n")
    
    def play(self, idx, time):
        # Plays an arm given index and time

        return self.data[idx][time]

def gen_adv_seq(rng, horizon, c):
    sequence = np.zeros(horizon, dtype=int)
    
    sequence[0] = rng.binomial(1, 0.5)
    
    for t in range(1, horizon):
        prob_stay = (1 + c) / 2
        
        stay = rng.binomial(1, prob_stay)
        
        if stay:
            sequence[t] = sequence[t-1]
        else:
            sequence[t] = 1 - sequence[t-1] # Flip the bit
            
    return sequence.tolist()