import numpy as np
from IPython.display import clear_output

SPINNER = ['/', '-', '\\', '|']

def eval_adv(model, algo):

    regrets = []
    for t in range(model.T):
        frame = SPINNER[t % len(SPINNER)]
        print(f'\r{frame} Running... Epoch {t+1}/{model.T}', end='', flush=True)

        choice = algo.step()
        best, rewards = model.play(t)
        algo.observe(rewards)

        regrets.append(rewards[best] - rewards[choice])
    
    cum_reg = np.cumsum(regrets).tolist()
    return cum_reg

def eval_stoch(model, algo):

    regrets = []
    for t in range(model.K):
        best, rewards = model.play(t)
        algo.observe(rewards[t])

    for t in range(model.K, model.T):
        frame = SPINNER[t % len(SPINNER)]
        print(f'\r{frame} Running... Epoch {t+1}/{model.T}', end='', flush=True)

        choice = algo.step()
        best, rewards = model.play(t)
        algo.observe(rewards)

        regrets.append(rewards[best] - rewards[choice])
    
    cum_reg = np.cumsum(regrets).tolist()
    return cum_reg

    