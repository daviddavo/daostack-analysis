import numpy as np

# TODO: Testear que funciona
def lac(x):
    # Formula from https://en.wikipedia.org/wiki/Lorenz_asymmetry_coefficient
    x = np.sort(x)
    n = x.size
    mu = np.average(x)
    m = np.searchsorted(x, mu)-1 # where to insert mu to keep order
    
    if n <= 1:
        return np.NaN
    
    # print("m:", m, "n:", n)
    
    if x[m+1] == x[m]: return np.NaN
    assert m >= 0, "m can't be negative"
    
    delta = (mu - x[m]) / (x[m+1]-x[m])
    
    F_mu = m + delta / n
    L_mu = (sum(x[:m-1]) + delta * x[m]) / sum(x)
    
    return F_mu + L_mu