import numpy as np
from IPython.display import clear_output

SPINNER = ['/', '-', '\\', '|']

def eval_adv(model, algo):

    regrets = []
    for t in range(model.horizon):
        frame = SPINNER[t % len(SPINNER)]
        print(f'\r{frame} Running... Epoch {t+1}/{model.horizon}', end='', flush=True)

        step, reward = algo.step(t)
        rewards = []
        for i in list(model.data):
            rewards.append(model.data[i][t])
        
        reg = (max(rewards) - reward)
        regrets.append(reg)
    
    cum_reg = np.cumsum(regrets).tolist()
    return cum_reg

    