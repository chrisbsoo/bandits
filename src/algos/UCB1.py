import numpy as np

class simple_ucb1:
    def __init__(self, model):
        self.K = model.K
        self.T = model.T
        self.N = np.zeros(self.K)
        self.p = np.zeros(self.K)
        self.UCB = np.full(self.K, np.inf)
        self.model = model

    def observe(self, reward):
        self.N[self.i] += 1
        self.p[self.i] += (1 / self.N[self.i]) * (reward - self.p[self.i])

    def step(self, active):
        self.t += 1
        term = (2 * np.log(self.t) + 1) / (self.N[active] + 1)
        self.UCB[active] = self.p[active] + np.sqrt(term)
        self.i = active[np.argmax(self.UCB[active])]
        return self.i
    
    def reset(self):
        self.N = np.zeros(self.K)
        self.p = np.zeros(self.K)
        self.i = None
        self.t = 0
    
    def eval(self, n_runs=1, reg=True, wk_reg=True, sleeping=True, graph=True):
        from src.utils import SPINNER
        from src.utils import plot
        from src.utils import estimate_exponent

        all_regrets = []
        all_regrets_wk = []
        
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
                self.observe(rewards[choice])
                
                regrets.append(rewards[self.model.best] - rewards[choice])
                regrets_wk.append(np.max(rewards[active]) - rewards[choice]) # weak regret
            
            all_regrets.append(np.cumsum(regrets))
            all_regrets_wk.append(np.cumsum(regrets_wk))

        params = {
            "n_runs" : n_runs,
            "regs" : all_regrets if reg else None,
            "wk_regs" : all_regrets_wk if wk_reg else None,
            "sleeping" : self.model.A.copy() if sleeping else None
        }
        
        mean_regrets = np.array(all_regrets).mean(axis=0)

        reg_est = estimate_exponent(all_regrets)
        wk_reg_est = estimate_exponent(all_regrets_wk)
        
        if graph:
            plot(self.model, **params)
            print("Regret Score: ", reg_est)
            print("Weak Regret Score: ", wk_reg_est)
        else:
            return reg_est, wk_reg_est