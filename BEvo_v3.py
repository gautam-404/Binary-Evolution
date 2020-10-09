#!/usr/local/bin/python

from __future__ import print_function

from amuse.lab import units, Particles, BSE
import numpy as np
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


def read_data(file, data):
    if len(data) == 0:
        data = np.loadtxt(file, delimiter = None)
    else:
        data = np.append(data, np.loadtxt(file, delimiter = None), axis=0 )
    return data


def separate_data(data, length):
    M_wd_zams = []
    M_secondary_zams = []
    a_zams = []

    for i in range(0, length):
        if data[i,12] > data[i,13]:
            M_wd_zams.append( data[i, 15] )
            M_secondary_zams.append( data[i, 14] )
        else: 
            M_wd_zams.append( data[i, 14] )
            M_secondary_zams.append( data[i, 15] )
  
        a_zams.append( data[i, 16] |units.RSun) 
    
    return M_wd_zams, M_secondary_zams, a_zams 


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



def magnetic_braking(primary, dt, B, M_old, mb_type):
    w = primary.spin.value_in(units.none)
    if mb_type == 1:
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
        
        # if P_dot>0 and 1e-10<P_dot<1e30:
        #     pass
        # else:
        #     P_dot = 0

        P = P + P_dot*dt.value_in(units.yr)
        w = ( 2 * np.pi / (P) ) * 3.154e+7          ## per year
        # print(P_dot)
        # exit()

        # print(w_s)

        # if w_s<1:
        #     print("Spinning up")
        # else:
        #     print("Spinning down")
        
        primary.spin = w |units.none
        return w, w_s, P_dot


    if mb_type == 0:
        #### Rotating magnetic dipole magnetic braking
        w = primary.spin.value_in(units.none)
        P = ( 2 * np.pi / w ) * 3.154e+7  #seconds

        mu = B*primary.radius.value_in(units.cm)**3 / 2         ## G cm^3
        
        P_dot = (B * np.sin(alpha)/ 3.1782086e+19)**2 / P
        # print(P_dot)
        # P_dot = -1e-22
        # P_dot = (B / 3.1782086e+19)**2 / P

        P = P + P_dot*dt.value_in(units.s)
        w = ( 2 * np.pi / ( P) ) * 3.154e+7
            
        primary.spin = w |units.none
        return w




def main_evolution(real_time, current_time, primary, secondary, B, printing, recycling, dt, MSP):
    secondary = stars[1]
    primary = stars[0]
    M_secondary_limit = 0.001
    dt_orig = dt
    n_wtf = 1
    n_bu = 1
    B_orig = B
    w_s = 0


    while primary.mass.value_in(units.MSun)>0 and real_time<=15e9:

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
        if primary.stellar_type.value_in(units.stellar_type) == 15 or secondary.stellar_type.value_in(units.stellar_type) == 15:
            # print("Massless SN")
            # input()
            if dt>1e5|units.yr:
                current_time -= dt
                real_time -= dt.value_in(units.yr)
                dt = dt/10
                code.particles.remove_particles(stars)
                stars_copy(primary_old, secondary_old)
                secondary = stars[1]
                primary = stars[0]
                code.binaries.remove_particles(binaries)
                binary_copy(current_time, a_old, 0)
                continue
            elif primary.stellar_type.value_in(units.stellar_type) == 15:
                break
            elif secondary.stellar_type.value_in(units.stellar_type) == 15 and primary.stellar_type.value_in(units.stellar_type) != 13:
                break
            elif secondary.stellar_type.value_in(units.stellar_type) == 15 and primary.stellar_type.value_in(units.stellar_type) == 13:
                dt = dt_orig
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


         ##### Magnetic Braking #####    
        if primary.stellar_type.value_in(units.stellar_type) == 12 and primary_old_type == 12:     ##WD
            if M_dot == 0:
                f = primary.radius.value_in(units.RSun) / 1.4e-5
                w = magnetic_braking(primary, dt, B/f, M_old, mb_type=0)

        if primary.stellar_type.value_in(units.stellar_type) == 13 and primary_old_type == 13:     ##NS
            if M_dot >= 1e-11:                       ## MB Accretion-Torque Model-2019, Gittins and Andersson et. al. ::: Accretng Millisecond X-ray Pulsars (AMXPs)
                ## Employing MB of type 2 (polar accretion), torque used in BSE ignored
                primary.spin = w_old |units.none   ## before code.evolve()
                w, w_s, P_dot = magnetic_braking(primary, dt, B, M_old, mb_type=1)
                P = ( 2 * np.pi /w ) * 3.154e+7
                if P==np.nan or P==-np.inf or P==np.inf or P<0 or P_dot == 0:
                    primary.spin = w_old |units.none
                    w = magnetic_braking(primary, dt, B, M_old, mb_type=0)
            elif 0<= M_dot <1e-11:
                primary.spin = w_old |units.none
                w = magnetic_braking(primary, dt, B, M_old, mb_type=0)
        primary.spin = w |units.none
        w = primary.spin.value_in(units.none)
        P = ( 2 * np.pi /w ) * 3.154e+7    #seconds
        P_new = P


        ### Acretors reaching break-up periods or if something goes wrong
        w_breakup = np.sqrt(G*primary.mass.value_in(units.MSun)/primary.radius.value_in(units.km)**3)
        P_breakup = ( 2 * np.pi /(w_breakup) ) * 3.154e+7    #seconds
#         print(P_breakup)
        if P_new <= P_breakup and secondary.mass.value_in(units.MSun) >= M_secondary_limit:           ## NS gains mass (might not be realistic) and shrinks. 
            if recycling == True:                                                               ## Thus we have a new breakup period. Essentially a rotationally supported NS
                # code.particles.remove_particles(stars)
                # stars_copy(primary_old, secondary)
                # secondary = stars[1]
                # primary = stars[0]
                primary.spin = w_old |units.none
#                 w, w_s, P_dot = magnetic_braking(primary, dt, B, M_old, mb_type=1)
        elif P_new <= P_breakup and secondary.mass.value_in(units.MSun) < M_secondary_limit:
            # code.particles.remove_particles(stars)
            # stars_copy(primary_old, secondary)
            # secondary = stars[1]
            # primary = stars[0]
            primary.spin = w_old |units.none
#             w = magnetic_braking(primary, dt, B, M_old, mb_type=0)
        primary.spin = w |units.none
        w = primary.spin.value_in(units.none)
        P = ( 2 * np.pi /w ) * 3.154e+7    #seconds
        P_new = P





        P_dot = (P_new - P_old) / dt.value_in(units.s)
        M_dot = ( primary.mass.value_in(units.MSun) - M_old )/dt.value_in(units.yr)   # Mass capture by primary per year
        M_loss = ( secondary.mass.value_in(units.MSun) - M_old_secondary )/dt.value_in(units.yr)   # Mass lost by secondary per year

        ### Spin-down Energy for Pulsars
        if primary.stellar_type.value_in(units.stellar_type) == 13:
            r = primary.radius 
            I = (0.4)*(primary.mass*r**2)
            I = I.value_in(units.g * units.cm**2) 
            E_dot = 4 * np.pi**2 * I * (P_dot/P**3)     # g cm^2 / s^3 = ergs/s
        else:
            E_dot = 0

        
        if primary.stellar_type.value_in(units.stellar_type) == 13 and M_dot==0:
            eta = 0.1
            # L_gamma = 4.8e33 * ((B**2)/1e17) * (P / 3e-3)**-4 * (eta/0.1)
            # L_gamma = eta * E_dot

            f, alpha, beta = 0.0122, -2.12, 0.82                                    ##slot-gap two-pole caustic (TPC) Gonthier 2018
            L_gamma = 6.8172e35 * f * (P/1e-3)**alpha * (P_dot/1e-20)**beta            ## erg/s
#             L_gamma = 1.724e34 * f * (P/1e-3)**alpha * (P_dot/1e-21)**beta            ## eV/s
        else:
            L_gamma = 0

        
        ## X-ray luminosity from pulsar spin-down
        if E_dot>0 and primary.stellar_type.value_in(units.stellar_type) == 13:
            L_x_pulsar_spindown = 1e-3 * E_dot
        else:
            L_x_pulsar_spindown = 0


        ### LMXB luminosity
        if M_dot>0:
            R_sch = 2 * G * primary.mass.value_in(units.MSun) / c**2
            eta = 0.5 * R_sch / primary.radius.value_in(units.km)
            L_x = eta * G * primary.mass.value_in(units.MSun) * M_dot / primary.radius.value_in(units.km)
            L_x = L_x * 2e33 * (1e5)**2 / (3.154e+7)**3  # ergs/s
        else:
            L_x = 0

        L_x = L_x + L_x_pulsar_spindown

        if (M_dot!=0 or M_loss!=0) or (L_x != 0 or L_gamma!=0):
            if primary.stellar_type.value_in(units.stellar_type) == 13:
                write_evolution(real_time, primary, secondary, current_time, L_x, P, P_dot, M_dot, M_loss, B, L_gamma, w_s, AIC=True)
            elif primary.stellar_type.value_in(units.stellar_type) == 13 and primary_old_type == 12:
                write_evolution(real_time, primary, secondary, current_time, L_x, P, P_dot, M_dot, M_loss, B, L_gamma, w_s, AIC=2) 
            else:
                write_evolution(real_time, primary, secondary, current_time, L_x, P, P_dot, M_dot, M_loss, B, L_gamma, w_s, AIC=False) 
            
        #### To print the evolution
        if printing == True:
            print("M_dot_secondary: %e, " %M_loss, "M_dot_primary: %e, " %M_dot,"L_x: %e" %L_x, "P_dot: %e" %P_dot, "L_gamma: %e" %L_gamma)
            print_evolution(primary, secondary, current_time, real_time)
            

        ### Other checks
        if (primary.stellar_type.value_in(units.stellar_type) != primary_old_type) and (primary.stellar_type.value_in(units.stellar_type) == 13 and primary_old_type != 12):
            break
        if (not 9<primary.stellar_type.value_in(units.stellar_type)<14) and secondary.stellar_type.value_in(units.stellar_type)>12:
            break

    
    return real_time, current_time, primary, secondary, primary_old, secondary_old, a_old






def evolve_binary(B, real_time):
    global primary, secondary, stars
    secondary = stars[1]
    primary = stars[0]
    current_time = 0 |units.yr
    dt = 1e7 |units.yr

    global alpha
    alpha = np.random.randint(0, 100 + 1)*np.pi/(2*100)

    # recycling = False
    recycling = True

    write_evolution(real_time, primary, secondary, current_time, 0, 0, 0, 0, 0, 0, 0, 0, AIC=False)
    
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
            print("No AIC...")
            print_evolution(primary, secondary, current_time, real_time)
    


    dt=1e7|units.yr
    if AIC == True:
        MSP = True
        M_ns = primary.mass.value_in(units.MSun)
        M_wd = primary.initial_mass.value_in(units.MSun)
        print(M_wd, M_ns)
        e = (M_wd - M_ns)/(M_ns + secondary_old.mass.value_in(units.MSun) )
        a = a_old
        a = a*(1+e)

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

        real_time, current_time, primary, secondary, primary_old, secondary_old, a_old = main_evolution(real_time, current_time, primary, secondary, B, printing, recycling, dt, MSP)

    w = primary.spin.value_in(units.none)
    P = ( 2 * np.pi /w ) * 3.154e+7    #seconds


    ## Important, not possible
    if primary.stellar_type.value_in(units.stellar_type) == 15 and primary_old.stellar_type.value_in(units.stellar_type) == 13:
        AIC = False
    if primary.stellar_type.value_in(units.stellar_type) != 13:
        AIC = False

    if AIC == True: 
        fmsp.write("%f  \t\t" %(t_aic) )
        fmsp.write("%E  \t\t" %(P) )
        fmsp.write("%E  \t\t" %(B) )
        fmsp.write("\n")
        # print("\n \n \n ")
        print_evolution(primary, secondary, current_time, real_time)
    
    return AIC


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







def parallel_evolution(sliced_data, slice_length, x, B_slice, tr_slice, outdir):
        np.seterr(divide = 'ignore')
        np.seterr(invalid = 'ignore')

        M_wd_zams, M_secondary_zams, a_zams  = separate_data(sliced_data, slice_length)


        if not os.path.exists(outdir):
            os.mkdir(outdir)
        

        global fmsp
        fmsp = open( os.path.join( outdir, 'MSPs%i.dat' %x), 'w')
        f_aic = open( os.path.join( outdir, 'AIC_events%i.dat' %x), 'w')

        one = 0
        n_aic = 0
        
        for i in range(0, slice_length):
            global code
            code = BSE()            # initialise BSE
            code.parameters.binary_enhanced_mass_loss_parameter = 0    ## def = 0
            code.parameters.common_envelope_efficiency = 1                ##alpha, def=1
            code.parameters.common_envelope_binding_energy_factor = 0.5     ##lambda, def=0.5
            code.parameters.common_envelope_model_flag = 0                 ## def = 0
            code.parameters.Eddington_mass_transfer_limit_factor = 1    ##def = 1
            code.parameters.wind_accretion_factor = 1.5                   ## def = 1.5
            code.parameters.wind_accretion_efficiency = 1                ## def = 1
            code.parameters.Roche_angular_momentum_factor = -1               ##def = -1
            code.parameters.white_dwarf_IFMR_flag =  0           ##ifflag > 0 uses white dwarf IFMR, def = 0    
            code.parameters.white_dwarf_cooling_flag =  1        ##wdflag > 0 uses modified-Mestel cooling for WDs (0). (default value:1) 
            code.parameters.neutron_star_mass_flag = 1      ##  def =1, Belczynski 
            
            init_binary(M_wd_zams[i]|units.MSun, M_secondary_zams[i]|units.MSun, a_zams[i]/2, 0)    

            global ehist
            ehist = open( os.path.join( outdir, "EvoHist_%i,%i.dat" %(x,i)), 'w')
                
            global real_time
            AIC = evolve_binary(B_slice[i], tr_slice[i])

            if AIC == True:
                n_aic += 1
                for j in range(17):
                    f_aic.write(str(sliced_data[i,j]) + " ")
                f_aic.write("\n")
                print("Completed calculating AIC: ", i)

            ehist.close()
            one += 1

            ## delete if file is empty
            if os.stat( os.path.join( outdir, "EvoHist_%i,%i.dat" %(x,i)) ).st_size == 0:
                os.remove(os.path.join( outdir, "EvoHist_%i,%i.dat" %(x,i)))
            ###compress the evo_hist file
            elif zip_file == True:
                with open(os.path.join( outdir, "EvoHist_%i,%i.dat" %(x,i)), 'rt') as f_in:
                    with gzip.open(os.path.join( outdir, "EvoHist_%i,%i.dat.gz" %(x,i)), 'wt') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(os.path.join( outdir, "EvoHist_%i,%i.dat" %(x,i)))


            # ## uncomment for only AIC systems
            # if AIC != True:
            #     os.remove(os.path.join( outdir, "EvoHist_%i,%i.dat" %(x,i)))


            

            code.stop()
            if printing == True:
                input("Evolution done....continue to next?")
        
                
        print(one)
        print(n_aic)

        f_aic.close()
        fmsp.close()

        




def process_outfiles(outdir):
    MSP_files = glob.glob( os.path.join( outdir, "MSPs*.dat") )
    AIC_files = glob.glob( os.path.join( outdir, "AIC_*.dat") )


    f_MSP = open( os.path.join( outdir, "MSPs_final.dat"), "w")
    for tempfile in MSP_files:
        f = open(tempfile)
        for line in f.readlines():
            f_MSP.write(line)
    f_MSP.close()
        
    f_AIC = open( os.path.join( outdir, "AIC_final.dat"), "w")
    for tempfile in AIC_files:
        f = open(tempfile)
        for line in f.readlines():
            f_AIC.write(line)
    f_AIC.close()


    print("\n \n Number of NSs formed via AIC = ", len(np.loadtxt( os.path.join( outdir, "MSPs_final.dat"))))

    return



def f(z):
    A = -2.62e-2
    B = 0.384
    C = -8.42e-2
    return A*z**2 + B*z + C 
D = 3.254


def z(t):
    t0 = t_end
    k = t0**(2.0/3)     # today at z = 0, t = t0
    if t == 0:
        z = 1e10
    else:
        z = k*(t)**(-2.0/3) - 1
    return z


def sfh():
    ## SFR
    t = np.arange(0, t_end, dt)
    sfh = []
    for time in t:  
        rate = 10**(max(f(z(time)), 0)) - D
        if rate > 0:
            sfh.append(rate)
        else:
            sfh.append(0)

    return np.array(sfh)


def Nformed_at_t(dt, M_bulge, M_sim):
    rate = sfh()*dt/(M_sim)
    sfr = (rate/sum(rate)) * length
    t = np.arange(0, t_end, dt)
    tr = []
    l = 0
    for i in range(len(t)):
        if sfr[i]!=0:
            for j in range( int(round(sfr[i])) ):
                tr.append( t[i] )
                l += 1
    if l>length:
        tr = tr[0:length]
    else:
        t_maxsfr = t[np.argmax(sfr)]
        tr += [t_maxsfr]*(length-l)
    return tr




if __name__ == "__main__":
    dt = 1e7
    t_end = 15e9
    
    M_bulge = 0.91e10
    M_sim = 1.63e8

    G = 1.3218607e+26               # km**3 * MSun**(-1) * yr**(-2)
    M_ch = 1.38 
    c = 9.4605284e12                # km/yr
    R_solar = 695700 
    outdir = 'OutputFiles_test_v1'
    if os.path.exists(outdir):
            os.system("rm -rf %s" %outdir)
    # zip_file = True
    zip_file = False

    global printing
    printing = False
    # printing = True


    data = []
    # data = read_data("ZAMS_new/aic_10colA", data)
    # data = read_data("ZAMS_new/aic_10colB", data)
    # data = read_data("ZAMS_/Hyd_accr.dat", data)
    # data = read_data("ZAMS_/He_accr.dat", data)
    # data = read_data("AIC_final_updated.dat", data)
    # data = read_data("AIC_new_updated.dat", data)
    data = read_data("AIC_final.5.dat", data)

    length = np.shape(data)
    length = length[0]
    
    print("Evolving %i binary systems. \n" %length)
    # input("Press any key to continue...")


    # B = 10**np.random.normal(loc=9, scale=0.3, size=len(data))
    B = 10**np.random.normal(loc=8.5, scale=0.475, size=len(data))

    tr = Nformed_at_t(dt, M_bulge, M_sim)


    ncores = 1
    slice_len = []
    data_slice = []
    B_slice = []
    tr_slice = []
    for x in range(0, ncores):
        data_slice.append( data[ int((x)*length/ncores) : int((x+1)*length/ncores) ] )
        B_slice.append( B[ int((x)*length/ncores) : int((x+1)*length/ncores) ] )
        tr_slice.append( tr[ int((x)*length/ncores) : int((x+1)*length/ncores) ] )
        slice_len.append( np.shape(data_slice[x])[0] )

    executor = concurrent.futures.ProcessPoolExecutor( max_workers = ncores )    

    if ncores == 1:
        parallel_evolution(data_slice[0], slice_len[0], 0, B_slice[0], tr_slice[0], outdir)
    else:
        futures = []
        for x in range(0, ncores):
            futures.append( executor.submit( parallel_evolution, data_slice[x], slice_len[x], x, B_slice[x], tr_slice[x], outdir) )

        print(concurrent.futures.wait(futures))


    # process_outfiles(outdir)
    # import read_outdata

    


