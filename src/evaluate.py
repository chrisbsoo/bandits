import numpy as np

def eval_adv(model, algo):

    regrets = []
    for t in range(model.horizon):
        step, reward = algo.step(t)
        rewards = []
        for i in list(model.data):
            rewards.append(model.data[i][t])
        
        reg = (max(rewards) - reward)
        regrets.append(reg)
    
    cum_reg = np.cumsum(regrets).tolist()
    return cum_reg

    