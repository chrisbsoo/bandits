def naive_greedy(sum1, n1, sum2, n2, BIG, core_avail):
    m1 = (sum1 / n1) if n1 > 0 else BIG
    m2 = (sum2 / n2) if n2 > 0 else BIG
    if core_avail:
        choice = 1 if m1 >= m2 else 2
    else:
        choice = 2

    return choice

def greedy_etc(sum1, n1, sum2, n2, core_avail, explore_m):
    if core_avail and n1 < explore_m:
        choice = 1
    elif n2 < explore_m:
        choice = 2
    elif core_avail:
        m1 = sum1 / n1
        m2 = sum2 / n2
        choice = 1 if m1 >= m2 else 2
    else:
        choice = 2
    return choice