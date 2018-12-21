import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
# from scipy.stats import norm
import random



def read_data(file, data):
    if len(data) == 0:
        data = np.loadtxt(file, delimiter = None)
    else:
        data = np.append(data, np.loadtxt(file, delimiter = None), axis=0 )
    return data

#Sorting Data
def sort_data():
    stars = data[np.argsort((data[:,0]))]
    f = open('Sorted stars.txt','w')
    for i in range(0,length): 
        for j in range(0,14):
            f.write("%f  " %stars[i,j])
        f.write("\n")
    f.close()
    return stars

def wd_mass_to_radius(M_wd):
    c_RM = ( 0.01 * 6.957e5) / ( 0.5 )**(-1/3)
    return c_RM * ( np.power(M_wd,-1/3) )



def separate_data(file):
    R = []
    c_RM = ( 0.01 * 6.957e5) / ( 0.5 )**(-1/3)

    M_wd = []
    M_wd_end = []

    M_donor = []

    M_accr = []
    M_loss = [] 


    t_start = []
    t_end = []
    k = 0

    if file == "DATA":
        f = open('DATA.dat','w')
    elif file == "AIC":
        f = open('AIC_events.dat','w')
    for i in range(0, length):
        if data[i,12]>data[i,13]:
            M_wd.append( data[i,4] )
            M_donor.append( data[i,3] )
            M_wd_end.append( data[i,10] )
        else: 
            M_wd.append( data[i,3] )
            M_donor.append( data[i,4] )
            M_wd_end.append( data[i,9] )
        
        M_accr.append( np.abs( min( data[i,12], data[i,13] ) ) )
        M_loss.append( np.abs( max( data[i,12], data[i,13] ) ) )

        R.append( wd_mass_to_radius(M_wd[k]) )
        
        t_start.append( data[i,0] )
        t_end.append( data[i,6] )       

        k+=1   
            

        for j in range(0,14):
            f.write(str(data[i,j]) + " ")
        f.write("\n")

    f.close()
    
    return M_donor, M_wd, M_wd_end, M_accr, R, t_start, t_end


# AIC time
def tM_new(nM, M_accr, M_max, t_start, t_end):
    dt = 1e3
    t = t_start
    while nM < M_max and t < t_end:  
        nM += M_accr * dt
        t += dt
    
    # t = ((1.4 - nM)/M_accr) + t_start
    return t, nM



#  AIC Events , writing them in AIC_events.txt
def aic():
    nM = [0]*length
    nM_wd = []
    M_init = []
    M_ch = 1.38

    t_last = [0]*length
    t_aic = []
    taic = []
    f = open('AIC_events.dat','w')

    for i in range(0, length):  
        if M_accr[i] != 0 and ( data[i, 1] == 12 or data[i, 7] == 12 ):    
            t_last[i], nM[i] =  tM_new(M_wd[i], M_accr[i], M_ch, t_start[i]*1e6, t_end[i]*1e6)
            # print( nM_wd[i])
                
            if nM[i] >= M_ch:
                # print( nM[i])
                # print(t_last[i] - t_start[i]*1e6)

                t_aic.append(t_last[i])
                taic.append(t_last[i]/1e6)
                nM_wd.append(nM[i])
                M_init.append( M_wd[i] )

                for j in range(0,14):
                    f.write(str(data[i,j]) + " ")
                f.write("\n")
    f.close()
    
    fig = plt.figure(figsize=(12,8))
    plt.hist(taic, 100)
    plt.ylabel('N')
    plt.xlabel('AIC delay times (Myr)')
    fig.savefig("hist_AIC.pdf")
    return t_aic, nM_wd



# Finding spin at the time of AIC

def spin_aic_wd():
    w = []
    P = []
    p = []
    P_init = []
    M_ns = []
    nR = []
    G = 1.3218607e+26 #km^3 M_sun^-1 yr^-2

    for i in range(0,length):  
        t1 = t_start[i] * 1e6
        t2 = t_aic[i]

        M_ns.append(  (random.randint(60, 80)/100 ) * nM_wd[i]  )
        
        
        random.seed(7511)
        P_init.append(  (1 / (365 * 24 * 3600 )) * random.randint(18000, 180000)  )       #years        #seconds(18000 to 180000)=(5 hrs to 50 hrs )
        

        # dw =  ( 2*np.pi / P_init[i]) + ( (5/2) * np.sqrt( G * M_wd[i] / R[i]**3) 
        # * M_accr[i] * ( t2 - t1 ) / ( M_wd[i] + (M_accr[i] * t2) ) )
        
        nR.append( wd_mass_to_radius(nM_wd[i]) )

        # w0 = ( 2*np.pi / P_init[i])
        w0=0

        w.append( ( (5/3)* np.sqrt(G*nM_wd[i]/R[i]**5)) + ((M_wd[i] * w0/nM_wd[i])) - ( (5/3)* (M_wd[i]/nM_wd[i]) *np.sqrt(G*M_wd[i]/R[i]**5)) )

    
        P.append(  ( 2 * np.pi /( w[i]) ) * 3.154e+7  )  #seconds
        p.append( np.log10(P[i]) )

    #Hist plot of periods
    fig = plt.figure(figsize=(12,8))
    n, bins, patches = plt.hist(p,300)
    # (mu, sigma) = norm.fit(p)
    # y = mlab.normpdf( bins, mu, sigma)
    # l = plt.plot(bins, y, 'r--', linewidth=2)
    plt.ylabel('N')
    plt.xlabel('log10(Period_WD (s))')
    fig.savefig("hist_P_wd.pdf")

    return P, w, M_ns



# Finding of the post AIC NS
def spin_ns():
    
    w_ns = [0]*length
    P_ns = [0]*length
    G = 1.3218607e+26 # km^3 M_sun^-1 yr^-2
    c = 9.4605284e12  # km/yr

    random.seed(7511)
    
    for i in range(0,length):
        R_core = R[i] * 0.1
        r = ( (2 * G * M_ns[i])/(c)**2 ) * (random.randint(25, 35) )/10
        # w_ns[i] = ( (M_wd[i] / M_ns[i]) * ( R[i]/r )**2 ) * w_wd[i]
        w_ns[i] = ( (M_wd[i] / M_ns[i]) * ( R[i]/r )**2 ) * w_wd[i] * (    (M_ns[i]*R_core**2)  /  ( ( (M_wd[i]-M_ns[i])*(R[i]**5 - R_core**5)/(R[i]**3 - R_core**3) ) )   )
        # w_ns[i] =  ( (5*M_ns[i] - 2*M_wd[i]) * R[i]**2 * w_wd[i])  / (3 * M_ns[i] * r**2)


        # P_ns[i] = ( (M_ns[i] / M_wd[i]) * ( r/R[i] )**2 ) * P_wd[i] #seconds 
        P_ns[i] = ( 2 * np.pi / ( w_ns[i]) ) * 3.154e+7 #seconds
    
    #Hist plot of periods
    fig = plt.figure(figsize=(12,8))
    p = np.log10(P_ns)
    n, bins, patches = plt.hist(p,300)
    # (mu, sigma) = norm.fit(p)
    # y = mlab.normpdf( bins, mu, sigma)
    # l = plt.plot(bins, y, 'r--', linewidth=2)
    plt.ylabel('N')
    plt.xlabel('log10(Period (s) )    NS')
    fig.savefig("hist_P_ns.pdf")
    
    return P_ns, w_ns



# Finding spin of the post AIC recycled NS
def recycled_spin_ns():
    w_rns = w_ns.copy()
    P_rns = [0]*length
    nM_ns = M_ns.copy()
    t_last = [0]*length
    G = 1.3218607e+26 #km^3 M_sun^-1 yr^-2
    c = 9.4605284e12  # km/yr

    f = open('MSPs.dat','w')
    f.write("Period (ms) \t")
    f.write("Accretion start time WD (Myr) \t\t")
    f.write("AIC delay time (Myr) \t\t\t\t")
    # f.write("NS accretion end time (Myr) \t\t\t\t")
    f.write("Accretion endtime (from StarTrack data-file)(Myr) \n")
    # print("Recycled: \n")
    for i in range(0,length):
        t1 = t_aic[i]
        t2 = t_end[i] * 1e6

        random.seed(7511)
        
        if (t2 - t1) > 0:
            # print(t2 - t1)
            r = ( (2*G*M_ns[i])/(c)**2 ) * (random.randint(25, 35))/10
            t_last[i], nM_ns[i] =  tM_new(M_ns[i], 2.923161e-8, 3, t1, t2)

            w0 = w_ns[i]
            w_rns[i] =  ( (5/3)* np.sqrt(G*nM_ns[i]/r**5)) + ((M_ns[i] * w0/nM_ns[i])) - ( (5/3)** (M_ns[i]/nM_ns[i]) *np.sqrt(G*M_ns[i]/r**5)) 
            
            P_rns[i] = ( 2 * np.pi / ( w_rns[i]) ) * 3.154e+7 #seconds

        else:
            P_rns[i] = P_ns[i]     #sec

        
        f.write("%f  \t\t" %(P_rns[i]*1e3) )
        f.write("%f \t\t\t\t\t\t\t\t\t" %t_start[i] )
        f.write("%f \t\t\t\t\t\t\t\t\t" %(t_aic[i]/1e6 ))
        # f.write("%f \t\t\t\t\t\t\t\t\t" %(t_last[i]/1e6 ))
        f.write("%f \n " %(t_end[i]) )

    f.close()
    
    #Hist plot of period
    fig = plt.figure(figsize=(12,8))
    p = np.log10(P_rns)
    plt.hist(p, 50)
    plt.ylabel('N')
    plt.xlabel('log10(Period (s) )  Recycled NS')
    fig.savefig("hist_P_rns.pdf")
    
    return P_rns, w_rns









if __name__ == '__main__':
    data = []

    # data = read_data("newtest.txt", data)
    # data = read_data("hydacr-14cols.dat", data)
    # data = read_data("amcvn-14cols.dat", data)
    # data = read_data("data/HeAcrWDs.dat", data)
    data = read_data("data/HAcrWDs.dat", data)
    length = np.shape(data)
    length = length[0]



    data = sort_data()
    length = np.shape(data)
    length = length[0]


    M_donor, M_wd, M_wd_end, M_accr, R, t_start, t_end = separate_data("DATA")
    

    M_accr = np.divide( M_accr, 10 )
    data = []
    data = read_data("DATA.dat", data)
    length = np.shape(data)
    length = length[0]

    t_aic, nM_wd = aic()

    data = []
    data = read_data("AIC_events.dat", data)
    length = np.shape(data)
    length = length[0]
    M_donor, M_wd, M_wd_end, M_accr, R, t_start, t_end = separate_data("AIC")
    # fig = plt.figure(figsize=(12,8))
    # plt.hist(t_end, 300)
    # plt.ylabel('N')
    # plt.xlabel('AIC delay times (Myr)')
    # fig.savefig("hist_AIC__.pdf")
    data = []
    data = read_data("AIC_events.dat", data)
    length = np.shape(data)
    length = length[0]

    P_wd, w_wd, M_ns = spin_aic_wd()

    P_ns, w_ns = spin_ns()


    P_rns, w_rns = recycled_spin_ns()

    for i in range(0, length):
        print(P_wd[i])   #sec
        print(P_ns[i])  #sec
        print(P_rns[i])    #sec
        # print(P_rns[i])
        # print(P_ns[i]*1e3 - P_ns_today[i])
        # print(t_aic[i] - t_start[i]*1e6)
        # print(t_end[i] - t_start[i])
        print("\n")     