import numpy as np

TOTAL_CHANNELS = 1024


def timeseries_to_channels(timeseries):
    """
    Convert a timeseries of photon arrival channels to a vector of counts, with counts[i] the counts in channel i
    """
    return np.bincount(timeseries, minlength=TOTAL_CHANNELS)


def channels_to_timeseries(channels, seed=None):
    """
    Convert a vector of counts in channels to a timeseries of photon arrival channels
    """
    return np.repeat(np.arange(TOTAL_CHANNELS), channels).astype(int)


if __name__=='__main__':
    timeseries = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    channels = timeseries_to_channels(timeseries)
    print(channels)
    print(channels_to_timeseries(channels))