#!/usr/local/bin/python

from __future__ import print_function

from amuse.lab import units, Particles, BSE
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib import gridspec
import gzip
import multiprocessing as mp
import concurrent.futures
import time, sys, os, shutil
import glob
import copy
import random
from scipy.stats import lognorm
from tqdm import tqdm
import paramiko


def read_data(filename):
    host = "mash"
    port = 22
    username = ""
    password = ""
    ssh_client = paramiko.SSHClient() 

    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(host, port, username, password)


    sftp_client = ssh_client.open_sftp()
    
    data = np.load("/home/aashimas/anuj/"+filename, allow_pickle=True)
    data = data.f.arr_0
    M_sim = float(sum(data[:,0])+sum(data[:,1]))
    sftp_client.close()
    ssh_client.close()
    return data, M_sim




def write_evolution(real_time, primary, secondary, current_time, L_x, P, P_dot, M_dot, M_loss, B, L_gamma, w_s, AIC):
    ehist.write( "%09.4f \t" %(real_time/1e6) )
    ehist.write( "%09.4f \t" %(current_time.value_in(units.Myr)) )
    ehist.write( "%08.4f \t" %(binary.semi_major_axis.value_in(units.RSun)*2) )
    ehist.write( "%.8f \t %.9f \t %s \t " %((primary.mass.value_in(units.MSun)) , (primary.radius.value_in(units.RSun)),  (primary.stellar_type.value_in(units.stellar_type))) )
    ehist.write( "%.8e \t" %(M_dot) )      # Mass capture by primary per year
    ehist.write( "%.8f \t %.9f \t %s \t " %((secondary.mass.value_in(units.MSun)) , (secondary.radius.value_in(units.RSun)),  (secondary.stellar_type.value_in(units.stellar_type))) )
    ehist.write("%.8e \t" %M_loss)     
    ehist.write( "%.8e \t" %(L_x))       ## accretion powered X-ray luminosity
    ehist.write( "%.8e \t" %(P))       ## Period
    ehist.write( "%.8e \t" %(P_dot))   ## P_dot
    ehist.write( "%.8e \t" %(B))       ## B (Gauss)
    ehist.write( "%.8e \t" %(L_gamma))       ## L_gamma
    ehist.write( "%.4e \t" %(w_s))       ## w_s < 1 spin-up, else spin-down
    ehist.write( "%i \t" %int(AIC))      ## AIC or not
    ehist.write( "\n" )

    return None


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



def magnetic_braking(primary, dt, B, M_old):
    w = primary.spin.value_in(units.none)   ##yr^-1
    P = ( 2 * np.pi /w ) * 3.154e+7    #seconds
    g = 1.3218607e+26 |(units.km**3 * units.MSun**(-1) * units.yr**(-2))
    m_dot = ( primary.mass - (M_old|units.MSun) )/dt   # Mass capture by primary per year
    gauss = units.cm**(-1.0/2) * units.g**(1.0/2) * units.s**(-1)
    P = ( 2 * np.pi / w ) * 3.154e+7  
    b = B |gauss
    r = primary.radius 

    mu = b*r**3 / 2         ## G cm^3

    I = (0.4)*(primary.mass*r**2)
    I = I.as_quantity_in(units.g * units.cm**2) 
    # print(I)

    eps = 1.4
    r_A = ( (mu)**4 / (2*g*primary.mass * m_dot**2) )**(1.0/7.0)   
    r_A = r_A.as_quantity_in(units.km)       # km
    r_m = eps*r_A
    # r_c =  5e9 * P |units.cm              # corotation radius
    # print( r_m.value_in(units.km), r_c.value_in(units.km) )

    w_K_r_m = np.sqrt(g*primary.mass/r_m**3)        
    w_K_r_m = w_K_r_m.as_quantity_in(1/units.yr)  

    w_s = primary.spin.value_in(units.none) / w_K_r_m.value_in(units.yr**(-1)) 
    # print(w_s)               

    P_dot = - (1-w_s) * 8.1e-5 * np.sqrt(eps) * (primary.mass.value_in(units.MSun)/1.4)**(3.0/7.0) * (1e45/I.value_in(units.g*units.cm**2)) * (mu.value_in(gauss*units.cm**3)/1e30)**(2.0/7.0) * ( P*abs(m_dot.value_in(units.MSun/units.yr)/1e-9)**(3.0/7.0) )**2     ## s/yr
    
    P_dot = P_dot/3.154e+7    ##s/s
    P = P + P_dot*dt.value_in(units.s)
    w = ( 2 * np.pi / (P) ) * 3.154e+7          ## per year
    if P_dot == np.nan or P == np.nan:
        w = primary.spin.value_in(units.none)
        P = ( 2 * np.pi /w ) * 3.154e+7    #seconds
        P_dot = (B / 3.1782086e+19)**2 / P
        P = P + P_dot * dt.value_in(units.s)
        w = ( 2 * np.pi / P ) * 3.154e+7
        
         


    # if w_s<1:
    #     print("Spinning up")
    # else:
    #     print("Spinning down")

    primary.spin = w |units.none
    return P, w, w_s, P_dot





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



def wd_mass_to_radius(M_wd):
    # R =  0.01 * ( M_wd**(-1/3)) * R_solar
    R =  0.01 * ( M_wd**(-1/3)) 
    return R

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



def evolve_binary(B, real_time):
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
            print("AIC, NS formed!")
            print_evolution(primary, secondary, current_time, real_time)
    else:
        AIC = False
        if printing == True:
            print("No AIC...")
            print_evolution(primary, secondary, current_time, real_time)
            input("Move fwd?")
        else:
            pass
#             print("No AIC...")
#             print_evolution(primary, secondary, current_time, real_time)
    
    ## potential aic
    if primary.stellar_type.value_in(units.stellar_type) == 12:
        print("ONe but no AIC, yet..")
        print_evolution(primary, secondary, current_time, real_time)
        
    WD = False
    if primary.stellar_type.value_in(units.stellar_type) in [10,11,12] or secondary.stellar_type.value_in(units.stellar_type) in [10,11,12]:
        WD = True



    
#     print(primary.initial_mass.value_in(units.MSun))
    t_aic = real_time
    w = primary.spin.value_in(units.none)
    P_aic = ( 2 * np.pi /w ) * 3.154e+7    #seconds

    B = random.choice(B_sam)
    dt = 1e8|units.yr
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

#     if AIC == True: 
#         fmsp.write("%f  \t\t" %(t_aic) )
#         fmsp.write("%E  \t\t" %(P_aic) )
#         fmsp.write("%E  \t\t" %(P) )
#         fmsp.write("%E  \t\t" %(B) )
#         fmsp.write("\n")
        # print("\n \n \n ")
#         print_evolution(primary, secondary, current_time, real_time)
    
    return AIC, WD


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







def parallel_evolution(i):

        M_wd_zams, M_secondary_zams, a_zams, e_zams  = data[i,0], data[i,1], data[i,2], data[i,3]


        if not os.path.exists(outdir):
            try:
                os.mkdir(outdir)
            except:
                pass
        

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

        init_binary(M_wd_zams|units.MSun, M_secondary_zams|units.MSun, a_zams/2|units.RSun, e_zams)    

#             global ehist
#             ehist = open( os.path.join( outdir, "EvoHist_%i,%i.dat" %(x,i)), 'w')
        global ehist_arr
        ehist_arr = []

        global real_time
#             AIC = evolve_binary(B_slice[i], tr_slice[i])
        AIC, WD = evolve_binary(0, tr[i])

        if AIC == True or WD == True:
            n_aic += 1
            print("Completed simulating AIC/WD: ", i)
            np.savez_compressed(os.path.join( outdir, "EvoHist_%i" %(i)), ehist_arr)
            one += 1



        code.stop()
        if printing == True:
            input("Evolution done....continue to next?")
        
                
#         print(one)
#         print(n_aic)

        




def process_outfiles(outdir):
    MSP_files = glob.glob( os.path.join( outdir, "MSPs*.dat") )
    AIC_files = glob.glob( os.path.join( outdir, "AIC_*.dat") )


    f_MSP = open( os.path.join( outdir, "/MSPs_final.dat"), "w")
    for tempfile in MSP_files:
        f = open(tempfile)
        for line in f.readlines():
            f_MSP.write(line)
        f.close()
    f_MSP.close()
        
    f_AIC = open( os.path.join( outdir, "/AIC_final.dat"), "w")
    for tempfile in AIC_files:
        f = open(tempfile)
        for line in f.readlines():
            f_AIC.write(line)
        f.close()
    f_AIC.close()


    print("\n \n Number of NSs formed via AIC = ", len(np.loadtxt( os.path.join( outdir, "MSPs_final.dat"))))

    return



## Bulge
def f_bulge(z):
    A = -2.62e-2
    B = 0.384
    C = -8.42e-2
    D = 3.254
    return A*z**2 + B*z + C, D


## disk
def f_disk(z):
    A = -4.06e-2
    B = 0.331
    C = 0.338
    D = 0.771
    return A*z**2 + B*z + C, D



# def z(t):
#     t0 = t_end
#     k = t0**(2.0/3)     # today at z = 0, t = t0
#     if t == 0:
def z(t):
    t0 = t_end
#     k = (2/69.4)*1e9*3.154e7
    z = np.sqrt((28e9 - t)/t) -1
    return z


def sfh(b_d):
    ## SFR
    t = np.arange(0, t_end, dt)
    sfh = []
    for time in t: 
        if b_d == "Bulge":
            ft, D = f_bulge(z(time))
        elif b_d == "Disk":
            ft, D = f_disk(z(time))
        rate = 10**(max(ft, 0)) - D
        if rate >= 0:
            sfh.append(rate)
        else:
            sfh.append(0)
    return np.array(sfh)


def Nformed_at_t(dt, M_bulge, M_sim, SFR, l):
    rate = SFR*(M_sim/M_bulge)
    sfr = (rate/sum(rate)) * l
#     print(sum(sfr), l)
    t = np.arange(0, t_end, dt)
#     plt.semilogy(t/1e9, (rate/sum(rate)))
#     plt.semilogy(t/1e9, sfr)
    tr = []
    l = 0
    for i in range(len(t)):
        if sfr[i]==0 or sfr[i]==np.nan:
            pass
        else:
            for j in range( int(round(sfr[i])) ):
                tr.append( t[i] )
                l += 1
#     if l>length:
#         tr = tr[0:length]
#     else:
#         t_maxsfr = t[np.argmax(sfr)]
#         tr += [t_maxsfr]*int(length-l)
    return tr#         z = 1e10
#     else:
#         z = k*(t)**(-2.0/3) - 1
#     return z





if __name__ == "__main__":
    np.seterr(divide = 'ignore')
    np.seterr(invalid = 'ignore')
        
    dt = 1e7
    t_end = 15e9
    
    M_bulge = 2e10
#     M_sim = 1e8

    G = 1.3218607e+26               # km**3 * MSun**(-1) * yr**(-2)
    M_ch = 1.38 
    c = 9.4605284e12                # km/yr
    R_solar = 695700 
    outdir = 'OutputFiles_scratch'
    if os.path.exists(outdir):
            os.system("rm -rf %s" %outdir)
    zip_file = True
    # zip_file = False

    global printing
    printing = False
#     printing = True

    print("Reading input files...")
#     data = np.load("aic_systems.npz", allow_pickle=True)['arr_0']
#     data = np.load("Init_param.npz", allow_pickle=True)['arr_0']
#     data = np.load("Init_param_1e8.npz", allow_pickle=True)['arr_0']
#     data = np.load("Init_param_1e6.npz", allow_pickle=True)['arr_0']
    data, M_sim = read_data("Init_param_1e7.npz")


    length = len(data)
    
    print("Evolving %i binary systems. \n" %length)
    print("Total mass being evolved = %e MSun \n" %M_sim)
    # input("Press any key to continue...")

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
        

    print("Sampling birth times...")
    t = np.arange(0, t_end, dt)
    t_end = 16e9
    dt = 1e6
    # length = len(ehists)
    M_b = 1.55e10
    sfr_b = sfh("Bulge")
    tr_sam = Nformed_at_t(dt, M_b, M_sim, sfr_b, int(1e8))
    tr = random.choices(tr_sam,  k=length)
    


    print("\n \n Starting parallel evolution...")
    ncores = None
    if ncores == 1:
        for i in range(length):
            parallel_evolution(i)
    else:
        with tqdm(total=length) as pbar:
            pool = mp.Pool(ncores, maxtasksperchild=int(1e6))
            for aic in enumerate(pool.imap_unordered(parallel_evolution, range(length))):
                pbar.update()


#     process_outfiles(outdir)
    # import read_outdata

    


