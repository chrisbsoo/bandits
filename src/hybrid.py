import numpy as np

class gauss_hybrid_model:
    def __init__(self, rng=None, K=None, T=None, s=None, n=None, c=None, l0=None, l1=None):
        # parameters
        self.rng = rng
        self.K = K
        self.T = T
        
        # hyperparameters
        self.s = s.reshape(-1, 1)      # (K,) -> (K, 1) smoothing parameter per arm
        self.n = n.reshape(-1, 1)      # (K,) -> (K, 1) number of 1s per arm
        self.c = c.reshape(-1, 1)      # (K,) -> (K, 1) min consecutive 1-1 pairs
        self.l0 = l0.reshape(-1, 1)    # (K,) -> (K, 1) P(asleep -> awake)
        self.l1 = l1.reshape(-1, 1)    # (K,) -> (K, 1) P(awake -> asleep)
        
        # K x T adversary matrix (arm, time)
        # Z[i, t] = 1, 0
        self.Z = self.gen_Z()

        # K x T mask matrix (arm, time)
        # A[i, t] = 1, 0. 1 -> active, 0 -> inactive
        self.A = self.gen_A()

        self.ZA = self.Z.astype(float) # copy, set to float
        self.ZA[self.A == 0] = np.nan # boolean masking
        
        # K x T reward matrix (arm, time)
        # Y[i, t] = N(Z[i, t], s) or NaN (inactive arm)
        self.Y = self.gen_Y()
    
    def play(self, idx, time):
        return self.Y[idx, time]
    
    def gen_Y(self):
        # (K, T) and (K, 1)
        # Returns NaN if ZA[i, t] = NaN
        return self.rng.normal(self.ZA, self.s)
    
    def gen_A(self):
        # init 0 mask matrix of shape (K, T)
        A = np.zeros((self.K, self.T), dtype=int)
        
        # P(A_{i, 1} = 1) = 1 for all i
        A[:, 0] = 1 
        
        for t in range(1, self.T):
            # sample independent uniform random variables U_{i, t}
            U = self.rng.random((self.K, 1))
            
            # Apply the recurrence relation formula
            stay_awake = (A[:, t-1:t] == 1) * (U >= self.l1) # P(1->0)
            wake_up = (A[:, t-1:t] == 0) * (U < self.l0) # P(0->1)
            
            # combine states to get A_{i, t}
            A[:, t] = (stay_awake + wake_up).squeeze() # flattens redundant dimensions
        
        return A
    
    def gen_Z(self):
        NotImplemented