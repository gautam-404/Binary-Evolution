import numpy as np
import matplotlib.pyplot as plt
import random
from amuse.lab import *
from amuse.units import units
from amuse.datamodel import Particle, Particles
from amuse.support.console import set_printing_strategy

from amuse.community.mesa.interface import MESA


def read_data(file, data):
    if len(data) == 0:
        data = np.loadtxt(file, delimiter = None)
    else:
        data = np.append(data, np.loadtxt(file, delimiter = None), axis=0 )
    return data

#Sorting Data
def sort_data():
    stars = data[np.argsort(data[:,0])]
    return stars

def separate_data():
    R = []
    M_wd = []
    M_wd_end = []
    M_donor = []
    Mdot_accr = []
    Mdot_loss = [] 
    t_start = []
    t_end = []
    a_init = []
    M_wd_type = []
    M_donor_type = []

    for i in range(0, length):
        if data[i,12] > data[i,13]:
            M_wd.append( data[i,4] |units.MSun )
            M_donor.append( data[i,3]|units.MSun )
            M_wd_type.append( data[i, 2] )
            M_donor_type.append( data[i, 1] )
        else: 
            M_wd.append( data[i,3]|units.MSun )
            M_donor.append( data[i,4] |units.MSun)
            M_wd_type.append( data[i, 1] )
            M_donor_type.append( data[i, 2] )
  
        Mdot_accr.append( np.abs( min( data[i,12], data[i,13] ) ) |units.MSun/units.Myr )
        Mdot_loss.append( - np.abs( max( data[i,12], data[i,13] ) ) |units.MSun/units.Myr )
        
        t_start.append( data[i,0] * 1e6  )
        t_end.append( data[i,6] * 1e6 )
        a_init.append( data[i,5])      
    
    return M_wd, M_donor, Mdot_accr, Mdot_loss, t_start, M_wd_type, M_donor_type, t_end, a_init


def wd_mass_to_radius(M_wd):
    # c_RM = ( 0.01 * 6.957e5) / ( 0.5 )**(-1/3)
    # return c_RM * ( M_wd**(-1/3))
    R =  0.01 * ( M_wd**(-1/3)) * R_solar
    return R

def spin_up(nM, M, R, w0):
    w = ( (5/3)* np.sqrt(G*nM/R**5)) + ((M * w0/nM)) - ( (5/3)** (M/nM) *np.sqrt(G*M/R**5))
    return w

def orbit(M1, M2, a_prevstep, Mdot_accr, Mdot_loss, dt):
    # a_temp = a_prevstep * ( (M1 * M2) / ( (nM+ M_disk) * M_donor) ) **2           # Mass conserved

    adot_massloss = 2*a_prevstep * ( ((Mdot_loss+Mdot_accr)*(M2-M1)/(M1*(M1+M2))) - ((Mdot_loss/M2) + (Mdot_accr/M1)) )   # Mass Lost from the system
    adot_grav = - (64/5) * (G**3 * M1*M2*(M1+M2)) / (c**5 * (a_prevstep*R_solar)**3)
    a = a_prevstep + (adot_grav/R_solar)*dt + adot_massloss*dt
    P = (2 * np.pi/ np.sqrt( G * (M1+M2)/ (a*R_solar)**3) ) * 365  #days
    return a


def donor_radius(star_donor, mass_loss_yn, mdot, dt):
    # We have to switch off all wind mass loss for manual mass loss to work
    star_donor.parameters.AGB_wind_scheme = 0
    star_donor.parameters.RGB_wind_scheme = 0
    if mass_loss_yn == False:
        star.time_step = dt|units.yr
        star.evolve_one_step()
        print("evolved to:", star.age, "->", star.radius)
        print(star.stellar_type)
    else:
        star.mass_change = mdot
        star.time_step = dt|units.yr
        star.evolve_one_step()
    return star.radius

def initial_donor_evolution(star_donor, donor_type):
    while star_donor.stellar_type != (donor_type | units.stellar_type):
        star_donor.time_step = 5e6|units.yr
        star_donor.evolve_one_step()
        print("evolved to:", star_donor.age, "->", star_donor.radius)
        print(star_donor.stellar_type)
    return star_donor

def roche_radius(M1, M2, a):
    q = M2/M1
    R_r = a * (0.49*q**(2/3)) / ( 0.6*q**(2/3) + np.log(1+q**(1/3)) )
    return R_r

# Finding of the post AIC NS
def spin_ns_aic(nM_wd, R, M_ns, w_wd):
    for i in range(0,length):
        # R_core = wd_mass_to_radius(nM_wd) * 0.9
        R_core = wd_mass_to_radius(M_ns)
        r = ( (2 * G * M_ns)/(c)**2 ) * (random.randint(25, 35) )/10
        # w_ns = ( (nM_wd / M_ns) * ( R/r )**2 ) * w_wd 

        w_ns = ( (nM_wd / M_ns) * ( R/r )**2 ) * w_wd * (   (M_ns*R_core**2)  /  ( ( (nM_wd-M_ns)*(R**5 - R_core**5)/(R**3 - R_core**3) ) )  )
        P_ns = ( 2 * np.pi / ( w_ns) ) * 3.154e+7  #seconds
    return P_ns, w_ns, r

def time_steps(nM, star_donor, donor_type, Mdot_accr, nu, Mdot_loss, t_start, a, M_limit, k, t1):
    M_disk = 0
    Mdot_gain = nu * Mdot_accr
    t = t_start
    dt = 1e2
    M1 = nM + M_disk
    M2 = star_donor.mass
    a_min = ( donor_radius(star_donor, False, Mdot_loss, 1e-6) + (wd_mass_to_radius(nM)/R_solar) )
    while nM < M_limit and a[k]> a_min and M_donor >= 0.1 and t <= 13.7e9:
        if donor_radius(star_donor, True, Mdot_loss, dt) > roche_radius(M1, M2, a[k]):
            M_donor += (Mdot_loss * dt)
            M_disk += (Mdot_accr - Mdot_gain) * dt
            if M_disk > 0:
                nM += Mdot_gain * dt
            
            M1 = nM + M_disk
            M2 = M_donor 
            a.append( orbit(M1, M2, a[k], Mdot_accr, Mdot_loss, dt)) 
            t += dt
            t1.append( t1[k] + dt )
        
        elif donor_radius(star_donor, True, Mdot_loss, dt) < roche_radius(M1, M2, a[k]) and M_disk > 0:
            if M_disk - (Mdot_gain * dt) < 0:
                M_disk = 0
            else:
                M_disk += - (Mdot_gain * dt)
            nM += Mdot_gain * dt
            M1 = nM + M_disk
            M2 = M_donor 
            a.append( orbit(M1, M2, a[k], Mdot_accr, Mdot_loss, dt)) 
            t += dt
            t1.append( t1[k] + dt )
        else:   
            a.append( orbit(M1, M2, a[k], 0, 0, 1e6)) 
            t += 1e6
            t1.append( t1[k] + 1e6 )
        a_min = ( donor_radius(star_donor, False, Mdot_loss, 1e-6) + (wd_mass_to_radius(nM)/R_solar) )
        k += 1

    return nM, M_donor, a, k, t, t1, M1



def time_evolution(M, M_donor, Mdot_accr, Mdot_loss, wd_type, donor_type, t_start, a_init):
    nM = M
    M_ns = 0
    t = t_start
    t1 = [0]
    k = 0  #step : number
    a = [a_init]
    aic = False
    random.seed(7511)

    stev = MESA(redirection='none')
    star_donor = stev.particles.add_particle(Particle(mass=M_donor))
    star_donor = initial_donor_evolution(star_donor, donor_type)
    
    if wd_type == 12: 
        nM, star_donor, a, k, t, t1, M1 = time_steps(nM, star_donor, donor_type, Mdot_accr, 0.4, Mdot_loss, t_start, a, M_ch, k, t1)
        if nM>=M_ch:                        # AIC
            aic = True 
            t_aic = t
            k_aic = k

            # P_init = (1 / (365 * 24 * 3600)) * random.randint(18000, 180000)        #years        #seconds(18000 to 180000)=(5 hrs to 50 hrs )
            # w0 = ( 2*np.pi / P_init ) 
            w0 = 0
            R = wd_mass_to_radius(nM)
            M_ns =  (random.randint(95, 99)/100 ) * nM
            # M_ns = 1.37
            e = (nM - M_ns)/(M_ns + star_donor.mass)
            t1.append(t1[k] )
            a.append( a[k]*(1+e) )
            k += 1
            print(a[0])
            print(donor_type)
            w_wd = spin_up(nM, M, R, w0)
            P_wd = ( 2 * np.pi / (w_wd) ) * 3.154e+7    #seconds
            # plt.plot(t1,a)
            # plt.show()
            k_aic = k
    

        # if aic == True:               # NS Recycling 
            P_ns, w_ns, r = spin_ns_aic(nM, R, M_ns, w_wd)     # at aic
            nM = M_ns
            nM, star_donor, a, k, t, t1, M1 = time_steps(nM, star_donor, donor_type, 2.92316e-8, 0.8, Mdot_loss, t_aic, a, 3, k, t1)
            w_rns = spin_up(nM, M_ns, r, w_ns)
            P_rns = ( 2 * np.pi /(w_rns) ) * 3.154e+7    #seconds
            print(a[k])
            print("\n")
            fig = plt.plot(np.log10(t1[0:k_aic]), np.log10(a[0:k_aic]), color = 'orange')
            plt.plot(np.log10(t1[k_aic:k]), np.log10(a[k_aic:k]), color = 'blue')
            plt.scatter(np.log10(t1[k_aic]), np.log10(a[k_aic]))
            plt.show()
            fig.savefig("plot_orbitalseparation_%i.pdf" %(t_aic))
            fmsp.write("%f  \t\t" %(P_rns*1e3) )
            fmsp.write("%f \t\t\t\t\t\t\t\t\t" %(t_start/1e6) )
            fmsp.write("%f \t\t\t\t\t\t\t\t\t" %(t_aic/1e6))
            fmsp.write("%f \t\t\t\t\t\t\t\t\t" %(t/1e6))
            fmsp.write("\n")
            return t, nM, M_donor, a[k], w_wd, P_wd, w_ns, P_ns, w_rns, P_rns, aic
        else:
            return t, nM, M_donor, a[k], 0, 0, 0, 0, 0, 0, aic
    else:
        return t, nM, M_donor, a[k], 0, 0, 0, 0, 0, 0, aic












if __name__ == '__main__':
    G = 1.3218607e+26 #km^3 M_sun^-1 yr^-2
    M_ch = 1.38
    c = 9.4605284e12  # km/yr
    R_solar = 695700 #km

    data = []
    data = read_data("newtest.txt", data)
    # data = read_data("hydacr-14cols.dat", data)
    # data = read_data("amcvn-14cols.dat", data)
    # data = read_data("data/HeAcrWDs.dat", data)
    # data = read_data("data/HAcrWDs.dat", data)
    # data = read_data("data/He-WDtoNS.dat", data)
    # data = read_data("data/H-WDtoNS.dat", data)
    length = np.shape(data)
    length = length[0]
    data = sort_data()

    M_wd, M_donor, Mdot_accr, Mdot_loss, t_start, wd_type, donor_type, t_end, a_init = separate_data()

    fmsp = open('MSPs.dat', 'w')
    f_aic = open('AIC_events.dat', 'w')
    fmsp.write("Period (ms) \t")
    fmsp.write("Accretion start time WD (Myr) \t\t")
    fmsp.write("AIC delay time (Myr) \t\t\t\t")
    fmsp.write("NS accretion end time (Myr) \t\t\t\t \n")
    one = 0
    n_aic = 0
    t, M_primary, M_secondary, a_final, w_wd_final, P_wd_final, w_ns_final, P_ns_final, w_rns_final, P_rns_final = [], [], [], [], [], [], [], [], [], []
    for i in range(0, length):
        t_temp, nM, M_donor_final, a_final_temp, w_wd_temp, P_wd_temp, w_ns_temp, P_ns_temp, w_rns_temp, P_rns_temp, aic = time_evolution(M_wd[i], M_donor[i], Mdot_accr[i], Mdot_loss[i], wd_type[i], donor_type[i], t_start[i], a_init[i])
        if aic == True:    
            t.append(t_temp)
            M_primary.append(nM)
            M_secondary.append(M_donor_final)
            a_final.append(a_final_temp)
            w_wd_final.append(w_wd_temp)
            P_wd_final.append(P_wd_temp)
            w_ns_final.append(w_ns_temp)
            P_ns_final.append(P_ns_temp)
            w_rns_final.append(w_rns_temp)
            P_rns_final.append(P_rns_temp)
            n_aic += 1
            for j in range(0,14):
                f_aic.write(str(data[i,j]) + " ")
            f_aic.write("\n")
        if wd_type[i] == 12:
            one += 1
    print(one)
    print(n_aic)

    fmsp.close()
 
