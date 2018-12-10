import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import random



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
            M_wd.append( data[i,4] )
            M_donor.append( data[i,3] )
            M_wd_type.append( data[i, 2] )
            M_donor_type.append( data[i, 1] )
        else: 
            M_wd.append( data[i,3] )
            M_donor.append( data[i,4] )
            M_wd_type.append( data[i, 1] )
            M_donor_type.append( data[i, 2] )
  
        Mdot_accr.append( np.abs( min( data[i,12], data[i,13] ) ) )
        Mdot_loss.append( - np.abs( max( data[i,12], data[i,13] ) ) )
        
        t_start.append( data[i,0] * 1e6 )
        t_end.append( data[i,6] * 1e6 )
        a_init.append( data[i,5])      
    
    return M_wd, M_donor, Mdot_accr, Mdot_loss, t_start, M_wd_type, M_donor_type, t_end, a_init


def wd_mass_to_radius(M_wd):
    # c_RM = ( 0.01 * 6.957e5) / ( 0.5 )**(-1/3)
    # return c_RM * ( M_wd**(-1/3))
    R =  0.01 * ( M_wd**(-1/3)) * R_solar
    return R

def spin_up(nM, M, R, w0):
    w = ( (5/3)* np.sqrt(G*nM/R**5)) + ((M * w0/nM)) - ( (5/3)*(M/nM) *np.sqrt(G*M/R**5))
    return w

def orbit(M1, M2, a_prevstep, Mdot_accr, Mdot_loss, dt):
    # a_temp = a_prevstep * ( (M1 * M2) / ( (nM+ M_disk) * M_donor) ) **2           # Mass conserved

    adot_massloss = 2*a_prevstep * ( ((Mdot_loss+Mdot_accr)*(M2-M1)/(M1*(M1+M2))) - ((Mdot_loss/M2) + (Mdot_accr/M1)) )   # Mass Lost from the system
    adot_grav = - (64/5) * (G**3 * M1*M2*(M1+M2)) / (c**5 * (a_prevstep*R_solar)**3)
    a = a_prevstep + (adot_grav/R_solar)*dt + adot_massloss*dt
    P = (2 * np.pi/ np.sqrt( G * (M1+M2)/ (a*R_solar)**3) ) * 365  #days
    return a

def donor_radius(M2, donor_type):
    if  donor_type ==0 or donor_type ==1 or donor_type ==3:
        if M2>=1:
            if 1<=M2<1.1:
                k = 1.05 / (1.1)**(3/7)
            elif 1.1<M2<1.3:
                k = 1.2 / (1.3)**(3/7)
            elif 1.3<M2<1.7:
                k = 1.3 / (1.7)**(3/7)
            else:
                k = 1.7 / (2.1)**(3/7)
            R = k * M2**(3/7)
        else:
            if 0.93<=M2<1.1:
                k = 0.93 / (0.93)**(4/5)
            elif 0.78<=M2<0.93:
                k = 0.85/ (0.78)**(4/5)
            elif 0.69<=M2<0.78:
                k = 0.74 / (0.69)**(4/5)
            elif 0.47<=M2<0.69:
                k = 0.63/ (0.47)**(4/5)
            elif 0.21<=M2<0.47:
                k = 0.32 / (0.21)**(4/5)
            else:
                k = 0.13 / (0.1)**(4/5)
            R = k * M2**(4/5)
    
    elif donor_type==8:
        R_ZHe = 0.2391*M2**4.6 / (M2**4 + (0.162*M2**3) + 0.0065)
        L_ZHe = 15262*M2**10.25 / (M2**9 + (29.54*M2**7.5) + (31.18*M2**6) + 0.0469) 
        alpha = max(0,  0.85 - 0.08*M2)
        L_THe = L_ZHe * (1.45 + alpha)
        lam = 500 * (2+M2**5) / M2**2.5
        L = M2 * 1e4
        R = R_ZHe * (L / L_THe)**0.2 + 0.02*(np.exp(L/lam) - np.exp(L_THe/lam))
        # print(R)
    else:
        R = 5
    return R

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

        # print(nM_wd-M_ns)
        w_ns = ( (nM_wd / M_ns) * ( R/r )**2 ) * w_wd * (   (M_ns*R_core**2)  /  ( ( (nM_wd-M_ns)*(R**5 - R_core**5)/(R**3 - R_core**3) ) )  )
        P_ns = ( 2 * np.pi / ( w_ns) ) * 3.154e+7  #seconds
    return P_ns, w_ns, r

def time_steps(nM, M_donor, wd_type, donor_type, Mdot_accr, nu, Mdot_loss, t_start, a, M_limit, k, t1):
    M_disk = 0
    Mdot_gain = nu * Mdot_accr
    t = t_start
    dt = 1e3
    M1 = nM[k] + M_disk
    M2 = M_donor[k]
    merger = False
    if wd_type == 12:
        R = wd_mass_to_radius(nM[k])
    else:
        R = 10

    while nM[k] < M_limit and M_donor[k] >= 0.1 and t <= 13.7e9:
        if donor_radius(M2, donor_type) > roche_radius(nM[k], M2, a[k]):
            M_donor.append( M_donor[k] + (Mdot_loss * dt) )
            M_disk += (Mdot_accr - Mdot_gain) * dt
            if M_disk > 0:
                nM.append( nM[k]+ Mdot_gain * dt)
            
            M1 = nM[k] + M_disk
            M2 = M_donor[k]
            a.append( orbit(M1, M2, a[k], Mdot_accr, Mdot_loss, dt)) 
            t += dt
            t1.append( t1[k] + dt )
        
        elif donor_radius(M2, donor_type) < roche_radius(nM[k], M2, a[k]) and M_disk > 0:
            if M_disk - (Mdot_gain * dt) < 0:
                M_disk = 0
            else:
                M_disk += - (Mdot_gain * dt)
            nM.append( nM[k]+ Mdot_gain * dt)
            M_donor.append( M_donor[k] )
            M1 = nM[k] + M_disk
            M2 = M_donor[k]
            a.append( orbit(M1, M2, a[k], Mdot_accr, Mdot_loss, dt)) 
            t += dt
            t1.append( t1[k] + dt )
        else:   
            a.append( orbit(M1, M2, a[k], 0, 0, 1e7)) 
            nM.append( nM[k] )
            M_donor.append( M_donor[k] )
            t += 1e7
            t1.append( t1[k] + 1e7 )
        a_min = ( donor_radius(M2, donor_type) + R/R_solar )
        k += 1
        if a[k] < a_min:
            nM.append(nM[k] + M_donor[k])
            M_donor.append(0)
            t1.append( t1[k] + 1e5 )
            a.append( a[k]) 
            k+=1
            merger = True
            break

    return nM, M_donor, a, k, t, t1, M1, merger



def time_evolution(M, M_don, Mdot_accr, Mdot_loss, wd_type, donor_type, t_start, a_init):
    nM = [M]
    M_donor = [M_don]
    M_ns = 0
    t = t_start
    t1 = [0]
    k = 0  #step : number
    a = [a_init]
    aic = False
    random.seed(7511)
    
    if wd_type == 12 and donor_type==8: 
        nM, M_donor, a, k, t, t1, M1, merger = time_steps(nM, M_donor, wd_type, donor_type, Mdot_accr, 0.4, Mdot_loss, t_start, a, M_ch, k, t1)
        if nM[k]>=M_ch:                        # AIC
            aic = True 
            t_aic = t
            k_aic = k
            wd_type = 13

            P_orb_init = 2 * np.pi / np.sqrt( G*(nM[0] + M_donor[0]) / (a[0]*R_solar)**3 )  # years
            P_init = P_orb_init
            w0 = ( 2*np.pi / P_init ) 
            R = wd_mass_to_radius(nM[k])
            M_ns =  (random.randint(90, 99)/100 ) * nM[k]
            # M_ns = 1.26
            e = (nM[k] - M_ns)/(M_ns + M_donor[k])
            t1.append(t1[k] )
            a.append( a[k]*(1+e) )
            nM.append( M_ns )
            M_donor.append( M_donor[k] )
            k += 1
            print(a[0])
            print(donor_type)
            w_wd = spin_up(nM[k], nM[0], R, w0)
            P_wd = ( 2 * np.pi / (w_wd) ) * 3.154e+7    #seconds

    
        
        if aic == True:               # NS Recycling 
            P_ns, w_ns, r = spin_ns_aic(nM[k-1], R, M_ns, w_wd)     # at aic
            nM, M_donor, a, k, t, t1, M1, merger = time_steps(nM, M_donor, wd_type, donor_type, 2.92316e-8, 0.8, Mdot_loss, t_aic, a, 3, k, t1)
            w_rns = spin_up(nM[k], M_ns, r, w_ns)
            P_rns = ( 2 * np.pi /(w_rns) ) * 3.154e+7    #seconds
            print(a[k])
            print('\n')
            # plot_orbital_separation(t1, a, nM, M_donor, k, k_aic)
            fmsp.write("%f  \t\t" %(P_rns*1e3) )
            fmsp.write("%f \t\t\t\t\t\t\t\t\t" %(t_start/1e6) )
            fmsp.write("%f \t\t\t\t\t\t\t\t\t" %(t_aic/1e6))
            fmsp.write("%f \t\t\t\t\t\t\t\t\t" %(t/1e6))
            fmsp.write("\n")
            return t_aic, nM, M_donor, a[k], w_wd, P_wd, w_ns, P_ns, w_rns, P_rns, aic
        else:
            return t, nM, M_donor, a[k], 0, 0, 0, 0, 0, 0, aic
    else:
        return t, nM, M_donor, a[k], 0, 0, 0, 0, 0, 0, aic



def plot_orbital_separation(t1, a, nM, M_donor, k, k_aic):
    fig = plt.figure(figsize=(8,4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2]) 

    ax0 = plt.subplot(gs[1])
    # ax0.plot(np.log10(t1[0:k_aic+1]), a[0:k_aic+1], color = 'orange', label='Pre-AIC')
    ax0.scatter(np.log10(t1[k_aic]), a[k_aic], label = 'AIC')

    if a[k] < ( donor_radius(M_donor[k], donor_type) + (10/R_solar) ):
        ax0.plot(np.log10(t1[k_aic:k]), a[k_aic:k], color = 'blue', label = 'Post-AIC')
        ax0.scatter(np.log10(t1[k]), a[k], color='red', label = 'Merger')
    else:
        ax0.plot(np.log10(t1[k_aic:k]), a[k_aic:k], color = 'blue', label = 'Post-AIC')

    ax1 = plt.subplot(gs[0], sharex = ax0)   
    ax1.plot(np.log10(t1[k_aic:k]), nM[k_aic:k], color = 'mediumvioletred', label = "M_accretor")
    ax1.plot(np.log10(t1[k_aic:k]), M_donor[k_aic:k], '--', color = 'mediumvioletred', label = "M_donor")
    # ax1.plot(np.log10(t1[0:k]), nM[0:k], color = 'mediumvioletred', label = "M_accretor")
    # ax1.plot(np.log10(t1[0:k]), M_donor[0:k], '--', color = 'mediumvioletred', label = "M_donor")
    
    plt.setp(ax1.get_xticklabels(), visible=False)
    yticks = ax0.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    # ax0.set_xlabel('$\log_{10}$( Time since WD accretion first began (Myr) )')
    ax0.set_xlabel('$\log_{10}$( Time since AIC (Myr) )')
    ax0.set_ylabel('Orbital Separation a  ($R_\odot$)')
    ax1.set_ylabel('Mass ($M_\odot$)')
    
    ax0.legend()
    ax1.legend()
    plt.subplots_adjust(hspace=.05)
    plt.show()
    # fig.savefig("plot_orbitalseparation_%i.pdf" %(t_aic))








if __name__ == '__main__':
    G = 1.3218607e+26 #km^3 M_sun^-1 yr^-2
    M_ch = 1.38
    c = 9.4605284e12  # km/yr
    R_solar = 695700 #km

    data = []
    # data = read_data("newtest.txt", data)
    data = read_data("AIC_events_1.dat", data)
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
    t_aic, M_primary, M_secondary, a_final, w_wd_final, P_wd_final, w_ns_final, P_ns_final, w_rns_final, P_rns_final = [], [], [], [], [], [], [], [], [], []
    for i in range(0, length):
        t_temp, nM, M_donor_final, a_final_temp, w_wd_temp, P_wd_temp, w_ns_temp, P_ns_temp, w_rns_temp, P_rns_temp, aic = time_evolution(M_wd[i], M_donor[i], Mdot_accr[i], Mdot_loss[i], wd_type[i], donor_type[i], t_start[i], a_init[i])
        if aic == True:    
            t_aic.append(t_temp)
            # M_primary.append(nM)
            # M_secondary.append(M_donor_final)
            # a_final.append(a_final_temp)
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
    # print(P_wd_final)
    # print(P_ns_final)
    fig = plt.figure(figsize=(12,8))
    plt.hist(P_wd_final, 500)
    plt.ylabel('N')
    plt.xlabel('log10(Period (s) )  WD')
    fig.savefig("hist_P_wd_mod.pdf")

    fig = plt.figure(figsize=(12,8))
    p = np.log10(P_rns_final)
    plt.hist(p, 100)
    plt.ylabel('N')
    plt.xlabel('log10(Period (s) )  Recycled NS')
    fig.savefig("hist_P_rns_mod.pdf")

    fig = plt.figure(figsize=(12,8))
    p = np.log10(P_ns_final)
    plt.hist(p, 100)
    plt.ylabel('N')
    plt.xlabel('log10(Period (s) )  NS')
    fig.savefig("hist_P_ns_mod.pdf")

    fmsp.close()
 
