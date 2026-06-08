import numpy as np

class gauss_hybrid_model:
    def __init__(self, rng=None, rngz=None, K=None, T=None, s=None, n=None, c=None, l0=None, l1=None):
        # parameters
        self.rngz = rngz
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
        Z = np.zeros((self.K, self.T), dtype=int)
        
        for i in range(self.K):
            ni = self.n[i]
            ci = self.c[i] * self.T
                
            # choose how many clusters (M) we can safely allow
            max_clusters = ni - ci

            # M clusters need at least M-1 zeros to separate them: ni + M - 1 <= T
            max_clusters = min(max_clusters, self.T - ni + 1)
                
            # uniformly choose num of clusters between 1 and max_clusters
            M = self.rngz.integers(1, max_clusters + 1)
            
            # distribute ni ones into M clusters (each size >= 1)
            dividers = np.sort(self.rngz.choice(ni - 1, size=M - 1, replace=False)) + 1
            dividers = np.concatenate(([0], dividers, [ni]))
            cluster_sizes = np.diff(dividers) # List of sizes summing to ni
            
            # distribute the remaining "free zeros" across the M+1 landing zones
            total_free_zeros = self.T - ni - (M - 1)
            
            # split free zeros into M+1 bins (can be 0 zeros in a bin)
            zero_dividers = np.sort(self.rngz.choice(total_free_zeros + M, size=M, replace=False))
            zero_dividers = np.concatenate(([0], zero_dividers, [total_free_zeros + M]))
            zero_bins = np.diff(zero_dividers) - 1 # How many zeros go in each gap
            
            # construct the row sequentially using our generated building blocks
            row = []
            for j in range(M):
                # add the zeros preceding this cluster
                row.extend([0] * zero_bins[j])
                # add the cluster of ones
                row.extend([1] * cluster_sizes[j])
                # add the mandatory separating zero (except after the very last cluster)
                if j < M - 1:
                    row.extend([0])
                    
            # add the final trailing zeros
            row.extend([0] * zero_bins[-1])
            
            Z[i, :] = np.array(row)
            
        return Z