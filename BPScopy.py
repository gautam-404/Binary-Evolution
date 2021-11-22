#!/usr/local/bin/python

import numpy as np
np.seterr(divide = 'ignore')
np.seterr(invalid = 'ignore')
import matplotlib.pyplot as plt
import istarmap
import multiprocessing as mp
import ray
import random
from tqdm import tqdm
import itertools
import BPS_SFH as SFH

from amuse.lab import Particles, units, BSE
import copy
import math
import sys, os

def release_list(a):
   del a[:]
   del a

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


def init_binary(M_primary, M_secondary, a, e):
    global stars 
    stars =  Particles(2)
    stars[0].mass = M_primary
    stars[1].mass = M_secondary

    global binaries
    binaries =  Particles(1)

    global binary
    binary = binaries[0]
    binary.semi_major_axis = a
    binary.eccentricity = e
    binary.child1 = stars[0]
    binary.child2 = stars[1]
    # we add the single stars first, as the binaries will refer to these
    code.particles.add_particles(stars)
    code.binaries.add_particles(binaries)

    global from_bse_to_model
    from_bse_to_model = code.particles.new_channel_to(stars)
    from_bse_to_model.copy()
        
    global from_bse_to_model_binaries
    from_bse_to_model_binaries = code.binaries.new_channel_to(binaries)
    from_bse_to_model_binaries.copy()

def evolve_binary(B, real_time, printing):
    global primary, secondary, stars
    secondary = stars[1]
    primary = stars[0]
    current_time = 0 |units.yr
    dt = 1e7 |units.yr
    AIC = False
    NS = False

    primary_old_type = primary.stellar_type.value_in(units.stellar_type)
    secondary_old_type = secondary.stellar_type.value_in(units.stellar_type)
    
    while primary.mass.value_in(units.MSun)>0 and real_time<=14e9:
        primary_old_type = primary.stellar_type.value_in(units.stellar_type)
        secondary_old_type = secondary.stellar_type.value_in(units.stellar_type)


        current_time += dt
        real_time += dt.value_in(units.yr)

        code.evolve_model( current_time )
        from_bse_to_model.copy()
        from_bse_to_model_binaries.copy()

        ehist_arr.append( [real_time/1e6, current_time.value_in(units.Myr), binary.semi_major_axis.value_in(units.RSun)*2, primary.mass.value_in(units.MSun), 
                               primary.radius.value_in(units.RSun), primary.stellar_type.value_in(units.stellar_type), 
                              secondary.mass.value_in(units.MSun), secondary.radius.value_in(units.RSun),  
                               secondary.stellar_type.value_in(units.stellar_type)] )

        if (primary_old_type == 12 and primary.stellar_type.value_in(units.stellar_type) == 13) or (secondary_old_type == 12 and secondary.stellar_type.value_in(units.stellar_type) == 13):
            AIC = True
            break
        if (primary.stellar_type.value_in(units.stellar_type) == 13 and primary_old_type!=13) or (secondary_old_type != 13 and secondary.stellar_type.value_in(units.stellar_type) == 13):
            NS = True
            break
        if primary.stellar_type.value_in(units.stellar_type) > 13 or secondary.stellar_type.value_in(units.stellar_type) > 13:
            break

    return AIC, NS


@ray.remote
def parallel_evolution(data, i, B, t_birth, printing):
    M1_zams, M2_zams, a_zams, e_zams  = data[0], data[1], data[2], 0

    outdir = os.path.expanduser('~')+"/OutputFiles"

    global code
    code = BSE()            # initialise BSE
    code.parameters.binary_enhanced_mass_loss_parameter = 0    ## def = 0
    code.parameters.common_envelope_efficiency = 1                ## alpha, def=1
    code.parameters.common_envelope_binding_energy_factor = -0.5     ## lambda, def=0.5
    code.parameters.common_envelope_model_flag = 0                 ## def = 0
    code.parameters.Eddington_mass_transfer_limit_factor = 1    ## def = 1
    code.parameters.wind_accretion_factor = 1.5                   ## def = 1.5
    code.parameters.wind_accretion_efficiency = 1                ## def = 1
    code.parameters.Roche_angular_momentum_factor = -1               ## def = -1
    code.parameters.white_dwarf_IFMR_flag =  0           ## ifflag > 0 uses white dwarf IFMR, def = 0
    code.parameters.white_dwarf_cooling_flag =  1        ## wdflag > 0 uses modified-Mestel cooling for WDs (0). (default value:1)
    code.parameters.neutron_star_mass_flag = 1      ## def = 1, Belczynski

    init_binary(M1_zams|units.MSun, M2_zams|units.MSun, a_zams|units.RSun, e_zams)    

    global ehist_arr
    ehist_arr = []

    global real_time
    AIC, NS = evolve_binary(B, t_birth, printing)

    if NS == True:
        # print("Completed simulating NS: ", i)
        np.savetxt(os.path.join( outdir, "EvoHist_%i" %(i)), ehist_arr)

    release_list(ehist_arr)

    code.stop()
    if printing == True:
        input("Evolution done....continue to next?")






if __name__ == "__main__":
    print("Reading the initital input parameters...")
    # filename = "Init_data_2e8.npz"
    filename = "Init_data_1e6.npz"
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

    outdir = os.path.expanduser('~')+"/OutputFiles"
    if not os.path.exists(outdir):
            try:
                os.mkdir(outdir)
            except:
                pass
    else:
        os.system("rm -rf "+outdir)
        os.mkdir(outdir)

    printing = False
    print("\n \n Starting parallel evolution...")
    ncores = 512
    ray.init(num_cpus=512)
    if ncores == 1:
        with tqdm(total=length) as pbar:
            for i in range(length):
                parallel_evolution(data[i], i, B_sam[i], tr[i], printing)
                pbar.update()
    else:
        # with mp.Pool(ncores) as pool:
            # iterable = list(zip(data, range(length), B_sam, tr, itertools.repeat(printing)))
        #     for _ in tqdm(pool.istarmap(parallel_evolution, iterable),
        #                     total=length):
        #         pass
        iterable = list(zip(data, range(length), B_sam, tr, itertools.repeat(printing)))
        result_ids = []
        with tqdm(total=length) as pbar:
            for i in range(length):
                result_ids.append(parallel_evolution.remote(data[i], i, B_sam[i], tr[i], printing))
                pbar.update()
        results = ray.get(result_ids)
        print(results)