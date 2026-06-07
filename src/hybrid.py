class gauss_hybrid_model:
    def __init__(self, rng, sigma, data):
        self.dist = rng.normal
        self.sigma = sigma
        self.rng = rng
        self.data = data

        self.horizon = len(self.data["0"])
        self.arm_size = len(self.data)
    
    def play(self, idx, time):
        mu = self.data[idx][time]
        return self.dist(mu, self.sigma ** 2)

