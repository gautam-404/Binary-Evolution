#!/usr/local/bin/python

import numpy as np
np.seterr(divide = 'ignore')
np.seterr(invalid = 'ignore')
import matplotlib.pyplot as plt
import istarmap
import multiprocessing as mp
import random
from tqdm import tqdm
import itertools
import os

import BPS_SFH as SFH
import BPS_evo_copy as evo



def read_data(filename):
    data = np.load(filename, allow_pickle=True)
    data = data.f.arr_0
    M_sim = sum(data[:,0])+sum(data[:,1])
    return data, M_sim

## Magnetic field strength distribution
def B_dist():
    print("B_dist...")
    mu = 8.21
    sigma = 0.21
    s = np.random.normal(mu, sigma, int(1e6))
    B_sim_chris = 10**s
    x = np.linspace(min(B_sim_chris), max(B_sim_chris), int(4e3))
    p = (np.sqrt(2*np.pi)*sigma)**-1 * np.exp( -(np.log10(x)-np.log10(10**mu))**2 /(2*sigma**2)  )
    B_sam = []
    for i in range(len(x)):
        B_sam += [x[i]]*int(len(x)*p[i])
    return B_sam



if __name__ == "__main__":
    print("Reading the initital input parameters...")
    filename = "Init_data_2e8.npz"
    if not os.path.isfile(filename):
        import BE_init
    data, M_sim = read_data(filename)
    length = len(data)
    n_sim = length
    
    print("Evolving %i binary systems. \n" %length)
    print("Total mass being evolved = %e MSun \n" %M_sim)

    
    B_sam = B_dist()


    dt = 1e7
    t_end = 14e9
    if os.path.isfile("tr.npz"):
        tr = np.load("tr.npz", allow_pickle=True)
        tr = tr.f.arr_0
    else:
        bd = input("\n What star frmation history do you want the stellar population to evolve with? The MW Bulge (enter b/B) or the MW Disk (enter d/D)...\n")
        if bd == 'b' or bd == 'B':
            tr = SFH.SFH(dt, t_end, n_sim, length, "Bulge")
        elif bd == 'd' or bd == 'D':
            tr = SFH.SFH(dt, t_end, n_sim, length, "Disk")
    # print(len(tr))

    printing = False
    print("\n \n Starting parallel evolution...")
    ncores = None
    if ncores == 1:
        with tqdm(total=length) as pbar:
            for i in range(length):
                evo.parallel_evolution(data[i], i, B_sam[i], tr[i], printing)
                pbar.update()
    else:
        # with tqdm(total=length) as pbar:
        #     pool = mp.Pool(ncores)
        #     iterable = list(zip(data, range(length), B_sam, tr, itertools.repeat(printing)))
        #     for aic in enumerate(pool.starmap(evo.parallel_evolution, iterable)):
        #         pbar.update()
        with mp.Pool(ncores) as pool:
            iterable = list(zip(data, range(length), B_sam, tr, itertools.repeat(printing)))
            for _ in tqdm(pool.istarmap(evo.parallel_evolution, iterable),
                            total=len(iterable)):
                pass