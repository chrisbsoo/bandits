import numpy as np

class simple_exp3:
    def __init__(self, model, gamma=None):
        self.K = model.K
        self.T = model.T
        self.model = model
        self.W = np.ones(self.K)

        if gamma is None:
            u = self.K * np.log(self.K)
            d = (np.e - 1) * self.T
            term = np.sqrt(u / d)
            self.gamma = min(1, term)
        else:
            self.gamma = gamma
    
    def observe(self, active, reward):
        r = reward / self.p[self.i]
        term = (self.gamma * r) / len(active)
        self.W[self.i] = self.W[self.i] * np.exp(term)
    
    def step(self, active):
        t1 = (1-self.gamma) * (self.W[active] / np.sum(self.W[active]))
        t2 = self.gamma / len(active)
        self.p = np.zeros(self.K)
        self.p[active] = t1 + t2

        self.p = self.p / self.p.sum()  # renormalisation for float-precision issues.

        assert np.isclose(sum(self.p), 1), f"Expected probabilities to sum up to 1. Got {sum(self.p)} instead."

        self.i = np.random.choice(active, p=self.p[active])
        return self.i
    
    def reset(self):
        self.W = np.ones(self.K)
        self.p = np.ones(self.K) / self.K
        self.i = None
    
    def eval(self, n_runs=1, reg=True, wk_reg=True, sleeping=True, graph=True):
        all_regrets = []
        all_regrets_wk = []

        from src.utils import SPINNER
        from src.utils import plot
        from src.utils import estimate_exponent
        
        for run in range(n_runs):
            frame = SPINNER[run % len(SPINNER)]
            
            self.reset()
            self.model.algo_reset()
            regrets = []
            regrets_wk = []

            bar1_ln = int((run / n_runs) * 10)
            bar1 = "X" * bar1_ln + "-" * (10 - bar1_ln)

            for t in range(self.model.T):
                frame1 = SPINNER[t % len(SPINNER)]

                bar_ln = int((t / self.model.T) * 10)
                bar = "X" * bar_ln + "-" * (10 - bar_ln)

                print(f'\r{frame}: [{bar1}] ({run} / {n_runs}) | {frame1}: [{bar}] ({t} / {self.model.T})', end='', flush=True)

                active, rewards = self.model.play(t)
                if len(active) == 0:
                    regrets.append(0)
                    regrets_wk.append(0)
                    continue
                choice = self.step(active)
                self.observe(active, rewards[choice])
                regrets.append(rewards[self.model.best] - rewards[choice]) # strong regret
                regrets_wk.append(np.max(rewards[active]) - rewards[choice]) # weak regret

            all_regrets.append(np.cumsum(regrets))
            all_regrets_wk.append(np.cumsum(regrets_wk))
        
        params = {
            "n_runs" : n_runs,
            "regs" : all_regrets if reg else None,
            "wk_regs" : all_regrets_wk if wk_reg else None,
            "sleeping" : self.model.A.copy() if sleeping else None
        }

        reg_est = estimate_exponent(all_regrets)
        wk_reg_est = estimate_exponent(all_regrets_wk)
        
        if graph:
            plot(self.model, **params)
            print("Regret Score: ", reg_est)
            print("Weak Regret Score: ", wk_reg_est)
        else:
            return reg_est, wk_reg_est






        
    





    


