import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

def hist_from_counts(counts:np.ndarray, **args):
    vec = []
    for i, count in enumerate(counts):
        vec.append([i for _ in range(count)])
    print (vec)
    plt.hist(vec, **args)


if __name__ == '__main__':
    counts = np.random.randint(1, 10, 30)
    plt.plot(counts)
    hist_from_counts(counts)