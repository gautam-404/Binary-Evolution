import numpy as np
from random import random
from tqdm import tqdm
import multiprocessing as mp

def init():
    # Initial masses in solar masses (M1 primary and M2 secondary). This range will always give you q<1
    M1_min = 5.0
    M1_max = 10.0
    M2_min = 0.5
    M2_max = 5.0 

    # periods in days:
    P_min = 10.0
    P_max = 10000.0

    # take the log in base 10:
    LP_min = np.log10(P_min)   
    LP_max = np.log10(P_max)  

    # Primary mass according to Kroupa's et al. slope of -2.3 for M>0.5 solar masses. The mass is generated from this power-law distribution from a uniform distribution rand(). See for instance https://mathworld.wolfram.com/RandomNumber.html.

    kroupa = -2.3
    M1 = ((M1_max**(kroupa+1.0)-M1_min**(kroupa+1.0))*random() +M1_min**(kroupa+1.0))**(1.0/(kroupa+1.0))  

    # Uniform (flat) distribution for secondary star:
        
    M2 = random()*(M2_max-M2_min)+M2_min                     
        
    # Orbit Period distribution is chosen to be flat in the log (as observed - I can send you a reference later):

    LogPeriod = random()*(LP_max-LP_min)+LP_min 
    Period = (10.0**LogPeriod)/365    ## in years
    G = 1.3218607e+26               # km**3 * MSun**(-1) * yr**(-2)
    a = (G*(M1+M2)*(0.5*Period/np.pi)**2)**(1/3) / 695700      ## RSun

    init_data.append([M1, M2, 2*a])
    # return M1, M2, a


if __name__ == "__main__":
    M_tot = float(input("Enter the total mass to be simulated (units MSun) \n"))
    # M_tot = 2e8
    manager = mp.Manager()
    init_data = manager.list()
    ncores = None
    l = int(M_tot/9.6)
    with tqdm(total=l) as pbar:
            pool = mp.Pool(ncores)
            for i in range(l):
                pool.imap_unordered(init(), 1)
                pbar.update()
    # print(init_data)
    init_data = np.array(init_data)
    print(sum(init_data[:,0])+sum(init_data[:,1]))
    np.savez_compressed("Init_data.npz", init_data)