import numpy as np
from scipy.signal import resample

def resample_ppg(data, new_length):
    return resample(data, new_length)
