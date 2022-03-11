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
    # filename = "Init_data_2e8.npz"
    filename = "Init_data.npz"
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
        bd = input("\n What star formation history do you want the stellar population to evolve with? The MW Bulge (enter b/B) or the MW Disk (enter d/D)...\n")
        if bd == 'b' or bd == 'B':
            tr = SFH.SFH(dt, t_end, n_sim, length, "Bulge")
        elif bd == 'd' or bd == 'D':
            tr = SFH.SFH(dt, t_end, n_sim, length, "Disk")
    # print(len(tr))

    #eccentricity
    # e = np.linspace(0,1)
    # f_e = []
    # for ee in e:
    #     if ee>0:
    #         f_e.append( 0.55/ee**(9/20) )
    #     else:
    #         f_e.append(0)
    # e_ = []
    # for i in range(len(e)):
    #     e_ += [e[i]]*int(len(e)*f_e[i]) 
    # ecc = np.random.choice(e_, len(data))

    #for 0 initial eccentricity
    ecc = [0]*len(data)

    outdir = os.path.expanduser('~')+"/OutputFiles"
    if not os.path.exists(outdir):
            try:
                os.mkdir(outdir)
            except:
                pass
    else:
        os.system("rm -rf "+outdir)
        os.mkdir(outdir)

    # printing = bool(input("Printing on? (True/False)"))
    printing = False
    print("\n \n Starting parallel evolution...")
    # ncores = int(input("Enter the number of parallel processes needed:"))
    ncores = 16
    if ncores == 1:
        with tqdm(total=length) as pbar:
            for i in range(length):
                evo.parallel_evolution(data[i], i, B_sam[i], tr[i], ecc[i], printing)
                pbar.update()
    else:
        with mp.Pool(ncores) as pool:
            iterable = list(zip(data, range(length), B_sam, tr, ecc, itertools.repeat(printing)))
            for _ in tqdm(pool.istarmap(evo.parallel_evolution, iterable),
                            total=length):
                pass

        # iterable = list(zip(data, range(length), B_sam, tr, itertools.repeat(printing)))
        # with tqdm(total=length) as pbar:
        #     for i in range(length):
        #         evo.parallel_evolution.remote(data[i], i, B_sam[i], tr[i], printing)
        #         pbar.update()
