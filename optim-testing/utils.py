import numpy as np


def estimate_exponent(cum_regrets):
    
    T = len(cum_regrets)
    t = np.arange(1, T+1)
    
    log_t = np.log(t)
    log_r = np.log(cum_regrets + 1e-10) 
    
    x = np.polyfit(log_t, log_r, 1)[0]   
    
    return x
