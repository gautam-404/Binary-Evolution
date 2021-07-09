#!/usr/local/bin/python

import numpy as np
from amuse.lab import units, Particles, BSE
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
from tqdm import tqdm

def read_data(filename):
    data = np.load(filename, allow_pickle=True)
    data = data.f.arr_0
    M_sim = sum(data[:,0])+sum(data[:,1])
    return data, M_sim




if __name__ == "__main__":
    filename = "Init_data_2e8.npz"
    data, M_sim = read_data(filename)



