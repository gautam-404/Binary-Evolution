import numpy as np
from amuse.lab import Particles, units, BSE
import copy
import math
import random
import sys, os

def release_list(a):
   del a[:]
   del a

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
        if (primary.stellar_type.value_in(units.stellar_type) == 13 and primary_old_type!=13) or (secondary_old_type != 13 and secondary.stellar_type.value_in(units.stellar_type) == 13):
            NS = True
        if primary.stellar_type.value_in(units.stellar_type) > 13 or secondary.stellar_type.value_in(units.stellar_type) > 13:
            # break
            pass

    return AIC, NS


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