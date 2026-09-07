import numpy as np

def bandits_ucb(sum1, n1, sum2, n2, t, core_avail, ucb_c):
    if core_avail and n1 == 0:
        choice = 1
    elif n2 == 0:
        choice = 2
    else:
        m2 = sum2 / n2
        ucb2 = m2 + ucb_c * np.sqrt(2.0 * np.log(t + 2.0) / n2)
        if core_avail:
            m1 = sum1 / n1
            ucb1 = m1 + ucb_c * np.sqrt(2.0 * np.log(t + 2.0) / n1)
            choice = 1 if ucb1 >= ucb2 else 2
        else:
            choice = 2

    return choice

def bandits_thompson(sum1, n1, sum2, n2, sigma, BIG, core_avail):
    if core_avail:
        if n1 > 0:
            m1 = sum1 / n1
            s1 = np.random.normal(m1, sigma / np.sqrt(n1))
        else:
            s1 = np.random.normal(0.0, BIG)
    if n2 > 0:
        m2 = sum2 / n2
        s2 = np.random.normal(m2, sigma / np.sqrt(n2))
    else:
        s2 = np.random.normal(0.0, BIG)
    if core_avail:
        choice = 1 if s1 >= s2 else 2
    else:
        choice = 2

    return choice
