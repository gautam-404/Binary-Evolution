import numpy as np
from amuse.lab import units, Particles, BSE
import copy
import math
import random
import sys, os

def init_star(i, star):
    global stars
    stars[i].mass = star.mass
    stars[i].radius = star.radius
    stars[i].spin = star.spin
    stars[i].stellar_type = star.stellar_type
    stars[i].core_mass = star.core_mass
    stars[i].core_radius = star.core_radius
    stars[i].epoch = star.epoch
    stars[i].initial_mass = star.initial_mass
    stars[i].luminosity = star.luminosity

def stars_copy(primary_old, secondary_old):
    global stars 
    stars =  Particles(2)
    init_star(0, primary_old)
    init_star(1, secondary_old)
    code.particles.add_particles(stars)



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



def binary_copy(current_time, a, e):
    global binaries
    binaries =  Particles(1)

    global binary
    binary = binaries[0]
    binary.semi_major_axis = a
    binary.eccentricity = e
    binary.age = current_time
    binary.child1 = stars[0]
    binary.child2 = stars[1]

    # we add the binaries which refer to old stars
    code.binaries.add_particles(binaries)

    global from_bse_to_model
    from_bse_to_model = code.particles.new_channel_to(stars)
    from_bse_to_model.copy()
        
    global from_bse_to_model_binaries
    from_bse_to_model_binaries = code.binaries.new_channel_to(binaries)
    from_bse_to_model_binaries.copy()


def print_evolution(primary, secondary, current_time, real_time):
    print("primary ->", primary.mass, primary.radius, primary.stellar_type)
    print("secondary ->", secondary.mass,  secondary.radius, secondary.stellar_type)
    print("Current time:", current_time.as_quantity_in(units.Myr), '|| Real time:', real_time/1e6)
    w_a = primary.spin.value_in(units.none)
    P_a = ( 2 * np.pi /w_a ) * 3.154e+7    #seconds
    print("P_a =", P_a)
    w_b = secondary.spin.value_in(units.none)
    P_b = ( 2 * np.pi /w_b ) * 3.154e+7    #seconds
    print("P_b =", P_b)
    print("Binary separation: %.20f RSun" %(binary.semi_major_axis.value_in(units.RSun)*2), ", Binary eccentricity:", binary.eccentricity)
    print("\n")
    sys.stdout.flush()
    return None


def spin_ns_aic(primary_old, primary):
    M_ns = primary.mass.value_in(units.MSun)
    M_wd = primary.initial_mass.value_in(units.MSun)
#     R = primary_old.radius.value_in(units.RSun)
    R = wd_mass_to_radius(M_wd)
#     R_core = primary.initial_radius.value_in(units.RSun)
    r = primary.radius.value_in(units.RSun)
    w_wd = primary_old.spin.value_in(units.none)
    w_ns = ( (M_wd / M_ns) * ( R/r )**2 ) * w_wd
#     print(w_wd, w_ns, M_wd, M_ns, R, r)
#     w_ns = ( (M_wd / M_ns) * ( R/r )**2 ) * w_wd * (   (M_ns*R_core**2)  /  ( ( (M_wd-M_ns)*(R**5 - R_core**5)/(R**3 - R_core**3) ) )  )
    P_ns = ( 2 * np.pi / ( w_ns) ) * 3.154e+7  #seconds
    return P_ns, w_ns

def wd_mass_to_radius(M_wd):
    # R =  0.01 * ( M_wd**(-1/3)) * R_solar
    R =  0.01 * ( M_wd**(-1/3)) 
    return R


def main_evolution(real_time, current_time, primary, secondary, B, printing, recycling, dt, MSP):
    secondary = stars[1]
    primary = stars[0]
    M_secondary_limit = 0.001
    dt_orig = dt
    n_wtf = 1
    n_bu = 1
    B_orig = B
    w_s = 0
    
    a_old = binary.semi_major_axis
    e_old = binary.eccentricity
    w_old = primary.spin.value_in(units.none)
    P_old = ( 2 * np.pi /w_old ) * 3.154e+7    #seconds
    primary_old_type = primary.stellar_type.value_in(units.stellar_type)
    secondary_old_type = secondary.stellar_type.value_in(units.stellar_type)
    M_old = primary.mass.value_in(units.MSun)
    M_old_secondary = secondary.mass.value_in(units.MSun)
    primary_old = copy.deepcopy(primary)
    secondary_old = copy.deepcopy(secondary)

    
    while primary.mass.value_in(units.MSun)>0 and real_time<=15e9:
        if primary.stellar_type.value_in(units.stellar_type) in [10,11,12]:
            dt = 1e8|units.yr
            
        if primary.stellar_type.value_in(units.stellar_type) == 12 and binary.semi_major_axis.value_in(units.RSun)<100:
            dt = 1e7|units.yr
            

        a_old = binary.semi_major_axis
        e_old = binary.eccentricity
        w_old = primary.spin.value_in(units.none)
        P_old = ( 2 * np.pi /w_old ) * 3.154e+7    #seconds
        primary_old_type = primary.stellar_type.value_in(units.stellar_type)
        secondary_old_type = secondary.stellar_type.value_in(units.stellar_type)
        M_old = primary.mass.value_in(units.MSun)
        M_old_secondary = secondary.mass.value_in(units.MSun)
        primary_old = copy.deepcopy(primary)
        secondary_old = copy.deepcopy(secondary)
    

        current_time += dt
        real_time += dt.value_in(units.yr)

    
        #### binary evolution only till a cutoff mass for the secondary
        if secondary.mass.value_in(units.MSun) >= M_secondary_limit:
            if recycling == False and primary_old_type == 13:
                pass
            else:
                code.evolve_model( current_time )
                from_bse_to_model.copy()
                from_bse_to_model_binaries.copy()

        M_dot = ( ( primary.mass - (M_old|units.MSun) )/dt ).value_in(units.MSun/units.yr)  # Mass capture by primary per year
        w = primary.spin.value_in(units.none)
        P = ( 2 * np.pi /w ) * 3.154e+7    #seconds
        P_bse = P
        w_bse = w

            
        
        ### Massless supernova and other evolutionary mistakes CHECK
        if primary.stellar_type.value_in(units.stellar_type) >9 or secondary.stellar_type.value_in(units.stellar_type) >9:
#             if dt>1e5|units.yr:
#                 current_time -= dt
#                 real_time -= dt.value_in(units.yr)
#                 dt = dt/10
#                 code.particles.remove_particles(stars)
#                 stars_copy(primary_old, secondary_old)
#                 secondary = stars[1]
#                 primary = stars[0]
#                 code.binaries.remove_particles(binaries)
#                 binary_copy(current_time, a_old, 0)
#                 continue
            if primary.stellar_type.value_in(units.stellar_type) == 15:
                break
            elif secondary.stellar_type.value_in(units.stellar_type) == 15 and primary.stellar_type.value_in(units.stellar_type) != 13:
                break
            elif primary.stellar_type.value_in(units.stellar_type) == 13:
#                 dt = dt_orig
                dt = 1e7 |units.yr
        elif primary.stellar_type.value_in(units.stellar_type) == primary_old_type or secondary.stellar_type.value_in(units.stellar_type) == secondary_old_type:
            dt = dt_orig
        if math.isnan(primary.radius.value_in(units.RSun)) or math.isnan(primary.mass.value_in(units.MSun)) or math.isnan(secondary.radius.value_in(units.RSun)) or math.isnan(secondary.mass.value_in(units.MSun)):
            break
        if real_time%dt_orig.value_in(units.yr)!=0 and dt==dt_orig:
            dt = dt_orig
            # current_time = np.ceil((current_time).value_in(units.Myr)/dt_orig.value_in(units.yr) )*10 |units.Myr
            current_time = np.ceil( current_time.value_in(units.Myr)/dt_orig.value_in(units.Myr)) * dt_orig.value_in(units.Myr) |units.Myr
            real_time = np.ceil( real_time/dt_orig.value_in(units.yr) ) * dt_orig.value_in(units.yr)
        
        
        ### AIC check, non-MSP run ###
        if MSP == False:
            if (primary_old_type == 12 and primary.stellar_type.value_in(units.stellar_type) == 13) or (secondary_old_type == 12 and secondary.stellar_type.value_in(units.stellar_type) == 13):
                # return secondary, primary, current_time, primary_old_type 
                break
                
            
        
        w = primary.spin.value_in(units.none)
        P = ( 2 * np.pi /w ) * 3.154e+7    #seconds
        P_new = P


        P_dot = (P_new - P_old) / dt.value_in(units.s)
        M_dot = ( primary.mass.value_in(units.MSun) - M_old )/dt.value_in(units.yr)   # Mass capture by primary per year
        M_loss = ( secondary.mass.value_in(units.MSun) - M_old_secondary )/dt.value_in(units.yr)   # Mass lost by secondary per year

        

        if (M_dot!=0 or M_loss!=0):
            ehist_arr.append( [real_time/1e6, current_time.value_in(units.Myr), binary.semi_major_axis.value_in(units.RSun)*2, primary.mass.value_in(units.MSun), 
                               primary.radius.value_in(units.RSun), primary.stellar_type.value_in(units.stellar_type), M_dot, 
                              secondary.mass.value_in(units.MSun), secondary.radius.value_in(units.RSun),  
                               secondary.stellar_type.value_in(units.stellar_type), M_loss, 0, P, P_dot, B, 0, 0, 0] )

            
        #### To print the evolution
        if printing == True:
#             print("M_dot_secondary: %e, " %M_loss, "M_dot_primary: %e, " %M_dot,"L_x: %e" %L_x, "P_dot: %e" %P_dot, "L_gamma: %e" %L_gamma)
            print("M_dot_secondary: %e, " %M_loss, "M_dot_primary: %e, " %M_dot)
            print_evolution(primary, secondary, current_time, real_time)
            

        ### Other checks
        if (primary.stellar_type.value_in(units.stellar_type) != primary_old_type) and (primary.stellar_type.value_in(units.stellar_type) == 13 and primary_old_type != 12):
            break               ### Only AIC NSs
        if (not 9<primary.stellar_type.value_in(units.stellar_type)<14) and secondary.stellar_type.value_in(units.stellar_type)>12:
            break

    return real_time, current_time, primary, secondary, primary_old, secondary_old, a_old




def evolve_binary(B, real_time, printing):
    global primary, secondary, stars
    secondary = stars[1]
    primary = stars[0]
    current_time = 0 |units.yr
    dt = 1e7 |units.yr

#     global alpha
#     alpha = np.random.randint(0, 100 + 1)*np.pi/(2*100)

    # recycling = False
    recycling = True

    ehist_arr.append( [real_time/1e6, current_time.value_in(units.Myr), binary.semi_major_axis.value_in(units.RSun)*2, primary.mass.value_in(units.MSun), 
                               primary.radius.value_in(units.RSun), primary.stellar_type.value_in(units.stellar_type), 0, 
                              secondary.mass.value_in(units.MSun), secondary.radius.value_in(units.RSun),  
                               secondary.stellar_type.value_in(units.stellar_type), 0, 0, 0, 0, B, 0, 0, 0] )
    
    real_time, current_time, primary, secondary, primary_old, secondary_old, a_old = main_evolution(real_time, current_time, primary, secondary, B, printing, recycling, dt, False)

    if primary.stellar_type.value_in(units.stellar_type) == 13 and primary_old.stellar_type.value_in(units.stellar_type) == 12:
        AIC = True
        t_aic = current_time.value_in(units.Myr)
        if printing == True:
            print("AIC, NS formed!")
            print_evolution(primary, secondary, current_time, real_time)
            input("Move fwd?")
    else:
        AIC = False
        if printing == True:
            print("No AIC...")
            print_evolution(primary, secondary, current_time, real_time)
            input("Move fwd?")
        
    WD = False
    if primary.stellar_type.value_in(units.stellar_type) in [10,11,12] or secondary.stellar_type.value_in(units.stellar_type) in [10,11,12]:
        WD = True

    
#     print(primary.initial_mass.value_in(units.MSun))
    t_aic = real_time
    w = primary.spin.value_in(units.none)
    P_aic = ( 2 * np.pi /w ) * 3.154e+7    #seconds

    dt = 1e7|units.yr
    if AIC == True or WD == True:
        Pns, wns = spin_ns_aic(primary_old, primary)
#         print(primary_old)
        primary.spin = wns |units.none
#         print(primary)
        M_ns = primary.mass.value_in(units.MSun)
        M_wd = primary.initial_mass.value_in(units.MSun)

        e = (M_wd - M_ns)/(M_ns + secondary_old.mass.value_in(units.MSun) )
        a = a_old*(1+e)

        # current_time -= dt 
        # real_time -= dt.value_in(units.yr)
        # code.particles.remove_particles(stars)
        # stars_copy(primary_old, secondary_old)
    
        code.particles.remove_particles(stars)
        stars_copy(primary, secondary)
        secondary = stars[1]
        primary = stars[0]
        code.binaries.remove_particles(binaries)
        binary_copy(current_time, a, e)

        real_time, current_time, primary, secondary, primary_old, secondary_old, a_old = main_evolution(real_time, current_time, primary, secondary, B, printing, recycling, dt, True)

        w = primary.spin.value_in(units.none)
        P = ( 2 * np.pi /w ) * 3.154e+7    #seconds
        
        if printing == True:
            print_evolution(primary, secondary, current_time, real_time)

    ## Important, not possible
#     if primary.stellar_type.value_in(units.stellar_type) == 15 and primary_old.stellar_type.value_in(units.stellar_type) == 13:
#         AIC = False
    if primary.stellar_type.value_in(units.stellar_type) != 13:
        AIC = False
        
    NS = False    
    if primary.stellar_type.value_in(units.stellar_type) == 13 or secondary.stellar_type.value_in(units.stellar_type) == 13:
        NS = True

    
    return AIC, WD, NS




def parallel_evolution(data, i, B, t_birth, ecc, printing):
    M1_zams, M2_zams, a_zams, e_zams  = data[0], data[1], data[2], ecc

    # outdir = os.path.expanduser('~')+"/OutputFiles"
    outdir = os.path.expanduser('~')+"/OutputFiles"
        

    one = 0
    n_aic = 0
    
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
    code.parameters.fractional_time_step_1 = 0.05    ## def = 0.05; MS phase
    code.parameters.fractional_time_step_2 = 0.01    ## def = 0.01; GB, CHeB, AGB, HeGB phase
    code.parameters.fractional_time_step_3 = 0.02    ## def = 0.02; HG, HeMS phase

    init_binary(M1_zams|units.MSun, M2_zams|units.MSun, a_zams|units.RSun, e_zams)    

    global ehist_arr
    ehist_arr = []

    global real_time
    AIC, WD, NS = evolve_binary(B, t_birth, printing)

    # if AIC == True or WD == True:
        # print("Completed simulating AIC/WD: ", i)
    np.savetxt(os.path.join( outdir, "EvoHist_%i" %(i)), ehist_arr)



    code.stop()
    if printing == True:
        input("Evolution done....continue to next?")
    
