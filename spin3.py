import numpy as np
import matplotlib.pyplot as plt
import random
# np.seterr(divide='ignore', invalid='ignore')
# from astropy import constants as const


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

def wd_mass_to_radius(M_wd):
    c_RM = ( 0.01 * 6.957e5) / ( 0.5 )**(-1/3)
    return c_RM * ( np.power(M_wd,-1/3) )



def separate_data(file):
    R = []
    M_wd = []
    M_wd_end = []
    M_donor = []
    Mdot_accr = []
    Mdot_loss = [] 
    t_start = []
    t_end = []
    a_init = []

    for i in range(0, length):
        if data[i,12] > data[i,13]:
            M_wd.append( data[i,4] )
            M_donor.append( data[i,3] )
        else: 
            M_wd.append( data[i,3] )
            M_donor.append( data[i,4] )
  
        
        Mdot_accr.append( np.abs( min( data[i,12], data[i,13] ) ) )
        Mdot_loss.append( - np.abs( max( data[i,12], data[i,13] ) ) )

        R.append( wd_mass_to_radius(M_wd[i]) )
        
        t_start.append( data[i,0] * 1e6 )
        t_end.append( data[i,6] * 1e6 )
        a_init.append( data[i,5])      
    
    return M_wd, M_donor, Mdot_accr, Mdot_loss, R, t_start, t_end, a_init


def spin_up(nM, M, R, w0):
    w = ( (5/3)* np.sqrt(G*nM/R**5)) + ((M * w0/nM)) - ( (5/3)** (M/nM) *np.sqrt(G*M/R**5))
    return w

def orbit(M1, M2, a_prevstep, Mdot_accr, Mdot_loss, dt):
    # a_temp = a_prevstep * ( (M1 * M2) / ( (nM+ M_disk) * M_donor) ) **2           # Mass conserved

    adot_massloss = 2*a_prevstep * ( ((Mdot_loss+Mdot_accr)*(M2-M1)/(M1*(M1+M2))) - ((Mdot_loss/M2) - (Mdot_accr/M1)) )   # Mass Lost from the system
    adot_grav = - (64/5) * (G**3 * M1*M2*(M1+M2)) / (c**5 * (a_prevstep*R_solar)**3)
    return a_prevstep + (adot_grav/R_solar)*dt + adot_massloss*dt

def donor_radius(M2):
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
    return R

def roche_radius(M1, M2, a):
    q = M2/M1
    R_r = a * (0.49*q**(2/3)) / ( 0.6*q**(2/3) + np.log(1+q**(1/3)) )
    return R_r

def time_steps(M, M_donor, Mdot_accr, nu, Mdot_loss, w0, R, t_start, a_init, M_limit):
    nM = M
    M_ns = 0
    M_disk = 0
    M_tot = M + M_donor
    Mdot_gain = nu * Mdot_accr
    dt = 1e2
    t = t_start
    t1 = [0]
    a = [a_init]
    k = 0
    M1 = nM + M_disk
    M2 = M_donor 
    while nM < M_limit and a[k]> 0 and M_donor >= 0.1:
        if donor_radius(M2) > roche_radius(M1, M2, a[k]):
            M1 = nM + M_disk
            M2 = M_donor 

            M_donor += (Mdot_loss * dt)
            M_disk += (Mdot_accr - Mdot_gain) * dt
            if M_disk >= 0:
                nM += Mdot_gain * dt
            
            a.append( orbit(M1, M2, a[k], Mdot_accr, Mdot_loss, dt)) 
            t += dt
            t1.append( t1[k] + dt )
        
        elif donor_radius(M2) < roche_radius(M1, M2, a[k]) and M_disk >= 0:
            M_disk += - (Mdot_gain * dt)
            nM += Mdot_gain * dt
            a.append( orbit(M1, M2, a[k], Mdot_accr, Mdot_loss, dt)) 
            t += dt
            t1.append( t1[k] + dt )
        else:   
            a.append( orbit(M1, M2, a[k], 0, 0, 1e7)) 
            t += 1e7
            t1.append( t1[k] + 1e7 )
        # print(M_disk)
        f1.write(str(M_disk) + "    " + str(t/1e6) + "  " + str(a[k]))
        f1.write("\n")
        k += 1

    if M_limit == M_ch and nM>=M_limit:    
        R = wd_mass_to_radius(nM)
        random.seed(7511)
        M_ns =  (random.randint(80, 90)/100 ) * nM

        e = (nM - M_ns)/(M_ns + M_donor)
        t1.append(t1[k])
        a.append( a[k]*(1+e) )
        # plt.plot(t1,a)
        # plt.show()
        print(a_init, a[k+1])
        # print("\n")
    
    elif M_limit == 4: 
        t1.append(t1[k])
        a.append(a[k])
        plt.plot(t1,a)
        plt.show()
        print(a_init, a[k+1])
    
    else:
        a.append(a[k])
    

    w = spin_up(nM, M, R, w0)
    P = ( 2 * np.pi /(w) ) * 3.154e+7    #seconds
    
    return t, nM, M_donor, M_ns, a[k+1], w, P



#  AIC Events , writing them in AIC_events.txt
def aic():
    M_wd_aic = []
    M_donor_aic = []
    a = []
    w0 = [0]*length
    w_wd = []
    P_init = [0]*length
    P_wd = []
    logP_wd = []
    t_aic = []
    taic = []
    M_ns = []


    f = open('AIC_events.dat','w')

    for i in range(0, length):  
        if Mdot_accr[i] != 0 and ( data[i, 1] == 12 or data[i, 7] == 12 ):    
            # random.seed(7511)
            P_init[i] = (1 / (365 * 24 * 3600 )) * random.randint(18000, 180000)        #years        #seconds(18000 to 180000)=(5 hrs to 50 hrs )
            w0[i] = ( 2*np.pi / P_init[i]) 
            
            t_aic_temp, nM_wd, nM_donor, M_ns_temp, a_temp, w_temp, P_temp = time_steps( M_wd[i], M_donor[i], Mdot_accr[i], 0.6, Mdot_loss[i], w0[i], R[i], t_start[i], a_init[i], M_ch)
        
            
            if nM_wd >= M_ch:
                t_aic.append( t_aic_temp )
                M_wd_aic.append( nM_wd ) 
                M_donor_aic.append( nM_donor ) 
                a.append( a_temp )
                w_wd.append( w_temp )
                P_wd.append( P_temp )
                M_ns.append(  M_ns_temp ) 
                # M_ns.append(  1.28 )
                taic.append( t_aic_temp/1e6 )
                logP_wd.append( np.log10(P_temp) )   #log (sec)

                for j in range(0,14):
                    f.write(str(data[i,j]) + " ")
                f.write("\n")
    f.close()
    # fig = plt.figure(figsize=(12,8))
    # plt.hist(t_aic, 50)
    # plt.ylabel('N')
    # plt.xlabel('AIC delay times (Myr)')
    # fig.savefig("hist_AIC.pdf")


    #Hist plot of periods
    # fig = plt.figure(figsize=(12,8))
    # plt.hist(logP_wd, 200)
    # plt.ylabel('N')
    # plt.xlabel('log10(Period_WD (s))')
    # fig.savefig("hist_P_wd.pdf")
    return t_aic, a, w_wd, P_wd, M_ns, M_wd_aic, M_donor_aic



# Finding of the post AIC NS
def spin_ns():
    
    w_ns = []
    P_ns = []


    random.seed(7511)
    
    for i in range(0,length):
        R_core = wd_mass_to_radius(M_wd[i]) * 1
        r = ( (2 * G * M_ns[i])/(c)**2 ) * (random.randint(25, 35) )/10
        w_ns.append( ( (M_wd[i] / M_ns[i]) * ( R[i]/r )**2 ) * w_wd[i] * (   (M_ns[i]*R_core**2)  /  ( ( (M_wd[i]-M_ns[i])*(R[i]**5 - R_core**5)/(R[i]**3 - R_core**3) ) )   ))
        P_ns.append( ( 2 * np.pi / ( w_ns[i]) ) * 3.154e+7 ) #seconds
    
    #Hist plot of periods
    # fig = plt.figure(figsize=(12,8))
    # p = np.log10(P_ns)
    # plt.hist(p, 200)
    # plt.ylabel('N')
    # plt.xlabel('log10(Period (s) )    NS')
    # fig.savefig("hist_P_ns.pdf")
    
    return P_ns, w_ns



# # Finding spin of the post AIC recycled NS
def recycled_spin_ns():
    w_rns = w_ns.copy()
    P_rns = [0]*length
    nM_ns = M_ns.copy()
    nM_donor = [0]*length
    t_last = [0]*length

    f = open('MSPs.dat','w')
    f.write("Period (ms) \t")
    f.write("Accretion start time WD (Myr) \t\t")
    f.write("AIC delay time (Myr) \t\t\t\t")
    # f.write("NS accretion end time (Myr) \t\t\t\t")
    f.write("Accretion endtime (from StarTrack data-file)(Myr) \n")
    # print("Recycled: \n")
    for i in range(0,length):
        random.seed(7511)
        r = ( (2*G*M_ns[i])/(c)**2 ) * (random.randint(25, 35))/10
        t_last[i], M_ns[i], M_donor[i], extra_garbage, a[i], w_rns[i], P_rns[i]= time_steps( M_ns[i], M_donor[i], 2.923161e-6 , 0.4, -4e-6, w_ns[i], r, t_aic[i], a[i], 4)            
        P_rns[i] = ( 2 * np.pi / ( w_rns[i]) ) * 3.154e+7 #seconds
        
        f.write("%f  \t\t" %(P_rns[i]*1e3) )
        f.write("%f \t\t\t\t\t\t\t\t\t" %(t_start[i]/1e6) )
        f.write("%f \t\t\t\t\t\t\t\t\t" %(t_aic[i]/1e6 ))
        # f.write("%f \t\t\t\t\t\t\t\t\t" %(t_last[i]/1e6 ))
        f.write("%f \n " %(t_end[i]/1e6) )

    f.close()
    
    #Hist plot of period
    # fig = plt.figure(figsize=(12,8))
    # p = np.log10(P_rns)
    # plt.hist(p, 200)
    # plt.ylabel('N')
    # plt.xlabel('log10(Period (s) )  Recycled NS')
    # fig.savefig("hist_P_rns.pdf")
    
    return P_rns, w_rns






if __name__ == '__main__':
    G = 1.3218607e+26 #km^3 M_sun^-1 yr^-2
    M_ch = 1.38
    c = 9.4605284e12  # km/yr
    R_solar = 695700 #km

    data = []
    # data = read_data("test.txt", data)
    # data = read_data("hydacr-14cols.dat", data)
    data = read_data("amcvn-14cols.dat", data)
    # data = read_data("data/HeAcrWDs.dat", data)
    # data = read_data("data/HAcrWDs.dat", data)
    length = np.shape(data)
    length = length[0]
    data = sort_data()

    M_wd, M_donor, Mdot_accr, Mdot_loss, R, t_start, t_end, a_init = separate_data("DATA")

    f1 = open('M_disk.txt','w')
    t_aic, a, w_wd, P_wd, M_ns, M_wd, M_donor = aic()

    length = np.shape(t_aic)
    length = length[0]
    data = []
    data = read_data("AIC_events.dat", data)
    M_wd0, M_donor0, Mdot_accr, Mdot_loss, R, t_start, t_end, a_init = separate_data("DATA")

    P_ns, w_ns = spin_ns()

    print(" \n \n \n ")
    P_rns, w_rns = recycled_spin_ns()
    f1.close()
    # for i in range(0, length):
        # print(M_ns[i], M_donor[i])
        # print(P_wd[i])  #sec
        # print(P_ns[i]*1e3)  #ms
        # print(P_rns[i]*1e3)  #ms
        # print("\n")     