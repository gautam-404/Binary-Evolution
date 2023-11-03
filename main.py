import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import istarmap
import multiprocessing as mp
from rich import print, prompt
from tqdm import tqdm
import itertools
import os
import BPS_SFH as SFH
import BPS_evo as evo
import BE_init as init

# Suppress runtime warnings
np.seterr(divide='ignore', invalid='ignore')

def read_data(filename):
    # data = np.load(filename, allow_pickle=True)
    # data = data.f.arr_0
    # M_sim = sum(data[:,0])+sum(data[:,1])
    # return data, M_sim
    data = pd.read_csv(filename)
    M_sim = sum(data['M1'])+sum(data['M2'])
    return data.values, M_sim

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

def runsfh(dt, t_end, M_sim, length):
    bd = prompt.Prompt.ask(f"What star formation history do you want the stellar population to evolve with? The MW Bulge (enter b/B) or the MW Disk (enter d/D)...", choices=["b", "B", "d", "D"])
    if bd == 'b' or bd == 'B':
        tr = SFH.sample_birth_times(dt, t_end, M_sim, length, "Bulge")
    elif bd == 'd' or bd == 'D':
        tr = SFH.sample_birth_times(dt, t_end, M_sim, length, "Disk")
    return tr

def create_input_data(filename, save=True):
    M_tot = float(input("Enter the total mass to be simulated (units MSun) \n"))
    binary_systems, total_mass = init.generate_binary_systems(M_tot, tolerance=1e7)
    print(f'Total mass simulated = {total_mass:.4e} MSun')
    if save:
        binary_systems.to_csv(filename, index=False)
    return binary_systems, total_mass

if __name__ == "__main__":
    filename = "input_data.csv"
    binary_systems, M_sim = None, None
    if not os.path.isfile(filename):
        binary_systems, M_sim = create_input_data(filename, False)
    elif prompt.Prompt.ask(f"Do you want to overwrite the existing input data file?", choices=["y", "n"]) == "y":
        binary_systems, M_sim = create_input_data(filename, False)
    else:
        print("Using existing input data file...")
        
    if binary_systems is None:
        print("\nReading the initial input parameters...\n")
        binary_systems, M_sim = read_data(filename)
    else:
        print("Using newly generated input data...")
    
    data = binary_systems.to_numpy()
    length = len(data)
    n_sim = length
    
    print("\nEvolving %i binary systems. \n" %length)
    print("\nTotal mass being evolved = %e MSun \n" %M_sim)

    B_sam = B_dist()

    dt = 1e7
    t_end = 14e9
    tr = runsfh(dt, t_end, M_sim, length)

    # if not os.path.isfile("tr.npz"):
    #     tr = runsfh(dt, t_end, M_sim, length)
    # else:
    #     x = input("\nStar-formation history data found from your prevoius run, do you wish to use it? (y/n)\n")
    #     if x == 'n' or x == 'N':
    #         tr = runsfh(dt, t_end, M_sim, length)
    #     else:
    #         tr = np.load("tr.npz", allow_pickle=True)
    #         tr = tr.f.arr_0

    #eccentricity distribution
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
    ecc = [0]*length

    outdir = "./OutputFiles"
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
    print("\n \n Starting parallel evolution...\n")
    # ncores = int(input("Enter the number of parallel processes needed:"))
    ncores = None
    if ncores == 1:
        with tqdm(total=length) as pbar:
            for i in range(length):
                evo.parallel_evolution(data[i], i, B_sam[i], tr[i], ecc[i], printing, outdir)
                pbar.update()
    else:
        with mp.Pool(ncores) as pool:
            iterable = list(zip(data, range(length), B_sam, tr, ecc, itertools.repeat(printing), itertools.repeat(outdir)))
            for _ in tqdm(pool.istarmap(evo.parallel_evolution, iterable),
                            total=length):
                pass
