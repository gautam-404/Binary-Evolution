import numpy as np
import glob
from tqdm import tqdm
import copy
import pickle
from amuse.lab import Particles, units

def read_eval(readdir):
    filenames = glob.glob(readdir+'/EvoHist*')
    ehists = np.array([])
    with tqdm(total=len(ehists)) as pbar:
        for i in range(len(filenames)):
            ehist_i = np.load(filenames[i], allow_pickle=True)
            ehist_i = ehist_i.f.arr_0
            ehists = np.append(ehists, ehist_i)
            pbar.update() 
    np.savez_compressed("ehists.npz")
    return ehists

def aic_index(ehists):
    print("Getting indices of AIC systems")
    aic_indices = []
    t_aic_all = []

    with tqdm(total=len(ehists)) as pbar:
        for i in range( len(ehists) ):
    #         onewd = False
            aic = False
            try:
                # k_onewd = list(ehists[i][:,5]).index(12)
                # k_ns = list(ehists[i][:,5]).index(13)
                aic_indices.append(i)
                # t_aic_all.append(ehists[i][k_ns,0])
                aic = True
            except ValueError:
                aic = False
            pbar.update()    
    return aic_indices


def magnetic_braking(P, dt, B, mass, m_dot, radius, a, AIC, alp, P_wd = 0, m_old = 1.44, r_old = 0.0145):
    flag = None
    P_old = P
    mmass = mass
    rradius = radius
    ww =  ( 2 * np.pi / P ) * 3.154e+7   ##yr^-1
    ddt = dt
    delKE_acc, delKE_prop, delKE_mdb = 0, 0, 0
    EdotGW = 0
    P_dot_final = 0
    P_dot_acc, P_dot_prop, P_dot_mdb, P_dot_GW = 0,0,0,0
    if AIC==True:
        w_s = 0
        G = 1.3218607e+26               # km**3 * MSun**(-1) * yr**(-2)
        RSun = 695700   ## km
        radius = rradius * RSun * 1e5     ## cm
        c = 2.998e10 ## cm/
        

        T = 1e11
        t_g = 47 * (mass/1.4)**-1 * (radius/10)**-4 * (P/1e-3)**6 / 3.154e+7    #syr
        tb = 2.7e11 * (mass/1.4) * (radius/10)**-1 * (P/1e-3)**2 * (T/1e9)**-6 / 3.154e+7     #yr
        ts = 6.7e7 * (mass/1.4)**(-5/4) * (radius/10)**(23/4) * (P/1e-3)**2 * (T/1e9)**2 / 3.154e+7     #yr
        tau = 1 / abs(tb**-1 + ts**-1 - abs(t_g)**-1)
        Jc = 3 * 1.635e-2 * alp**2 * mass*radius**2 / 2
        Jc_dot = - 2* Jc/tau
    
                      
        dt = tau   #yr
                      
        I = (0.4)*(mass*radius**2)      ##g cm^2
        Idot = (I - (0.4)*(m_old * (r_old*RSun*1e5)**2)) / dt    
        w_wd = ( 2 * np.pi / P_wd ) * 3.154e+7   ##yr^-1
        w = ( 2 * np.pi / P ) * 3.154e+7   ##yr^-1
        wdot = (w - w_wd) / dt
        wdot_tot = wdot
        w += wdot*dt/3.154e+7 
#         print(wdot)
#         m_coll_dot = abs(mass - m_old) / dt

        wdot = 3 * 1.635e-2 * alp**2 * w / (0.261*t_g)
        w += wdot*t_g/3.154e+7 
        Jdot = Idot*w + I*wdot + Jc_dot
        wdot = (Jdot/I) - (m_dot*w/mass) - (3*1.635e-2*alp**2*w/(I*tau))
        if 0.01<alp<1:
            wdot = (Jdot/I) * (1 - (3 * 1.635e-2 * alp**2 / (2*0.261)))**-1 - (m_dot*w/mass)
        wdot_tot += wdot
#         print(wdot)
        w += wdot*dt/3.154e+7 
        P = ( 2 * np.pi / w ) * 3.154e+7
        
        alp_dot = -alp*(1/tau + wdot_tot/(2*w) + m_dot/(2*mass))
        alp += alp_dot
        P_dot_GW = (P - P_old)/(ddt*3.154e+7)
        P_dot_final = P_dot_GW
#         print(P_old, P)
    else:       
        mdot_lim = 0 
        if m_dot > mdot_lim: ## MB Accretion-Torque Model-2019, Gittins and Andersson et. al. ::: Accretng Millisecond X-ray Pulsars (AMXPs
            g = 6.67408e-11 |(units.m**3 * units.kg**(-1) * units.s**(-2))
            gauss = units.cm**(-1.0/2) * units.g**(1.0/2) * units.s**(-1)
            b = B |gauss
            mass = mass |units.MSun
            r = rradius |units.RSun
            m_dot = m_dot |units.MSun/units.yr
            dt = dt |units.yr
            w = ( 2 * np.pi / P ) * 3.154e+7   ##yr^-1
            w_old = w |units.yr**-1

            mu = b*r**3         ## G cm^3

            I = (0.4)*(mass*r**2)
    #         I = I.as_quantity_in(units.g * units.cm**2) 
            # print(I)

            r_A = ( (mu)**4 / (2*g*mass * m_dot**2) )**(1.0/7.0)   
            r_A = r_A.as_quantity_in(units.km)       # km
            r_m = xi*r_A
            r_c = (g*mass/w_old**2)**(1.0/3.0) 

            w = ( 2 * np.pi / P ) * 3.154e+7
            w_K_r_m = np.sqrt(g*mass/r_m**3)        
            w_K_r_m = w_K_r_m.as_quantity_in(1/units.yr)  
            w_s = w / w_K_r_m.value_in(units.yr**(-1)) 
    #         P_old = P
            if r_m.value_in(units.km) < r_c.value_in(units.km):
                w_dot = m_dot*np.sqrt(g*mass*r_m)/I
                w = ww + w_dot.value_in(units.yr**-2)*dt.value_in(units.yr)
                P = ( 2 * np.pi / w ) * 3.154e+7
                P_dot = (P - P_old)/dt.value_in(units.s)
                P_dot_acc = P_dot
                P_dot_final = P_dot_acc
                flag = 1

                m =  mmass * 1.989e+33  ##g
                r =  rradius * 6.957e+10     ##cm
                I_old = (0.4)*(m*r**2)      ##g cm^2
                wn = ( 2 * np.pi / P ) * 3.154e+7
                delKE_acc = 0.5 * (I_old*wn**2 - I_old*ww**2)
#                 print(wn, P)

            if r_m.value_in(units.km) >= r_c.value_in(units.km):
                ww = ( 2 * np.pi / P ) * 3.154e+7
                P_dot = - (1-w_s) * 8.1e-5 * np.sqrt(xi) * (mass.value_in(units.MSun)/1.4)**(3.0/7.0) * (1e45/I.value_in(units.g*units.cm**2)) * (mu.value_in(gauss*units.cm**3)/1e30)**(2.0/7.0) * ( P_old*abs(m_dot.value_in(units.MSun/units.yr)/1e-9)**(3.0/7.0) )**2     ## s/yr
                P_dot = P_dot/3.154e+7   ## s/s
                P_dot_prop = P_dot
                P_dot_final = P_dot_prop
                P = P + P_dot*dt.value_in(units.s)
                Pn = P
                wn = ( 2 * np.pi / Pn )*3.154e+7
    #             w_dot = ( 2 * np.pi / Pn ) * 3.154e+7  -  ( 2 * np.pi / P_old ) * 3.154e+7
                flag = 2

                m =  mmass * 1.989e+33  ##g
                r =  rradius * 6.957e+10     ##cm
                I_old = (0.4)*(m*r**2)      ##g cm^2
                
                
                delKE_prop = 0.5 * (I_old*wn**2 - I_old*ww**2)
    #             print(delKE_prop/(1e7*3.154e+7)**2)

    #         Q22 = (1-w_s) * 4.2e37 * xi**0.25 * (mass.value_in(units.MSun)/1.4)**(3.0/14) * (mu.value_in(gauss*units.cm**3)/1e30)**(1.0/7.0) * (m_dot.value_in(units.MSun/units.yr)/1e-9)**(3.0/7.0) * (P_old**-1 / 500)**(-5.0/2)    ## g cm^2
    #         Pdot_GW = 1.4e-19 * (1e45/I.value_in(units.g*units.cm**2)) * (Q22/1e37)**2 * P**-3  ##s/yr
    #         Pdot_GW = Pdot_GW/3.154e+7   ## s/s
    #         P = P + Pdot_GW*dt.value_in(units.s)
    #         P_dot = P_dot + Pdot_GW
    #         w = ( 2 * np.pi / P ) * 3.154e+7


#         elif m_dot <= mdot_lim:
        ww = ( 2 * np.pi / P ) * 3.154e+7
        m = mmass * 1e33
        radius = radius * 695700 * 1e5     ## cm
        c = 2.998e10 ## cm/s
        I = (0.4)*(m*radius**2)      ##g cm^2
        mu = B*radius**3         ## G cm^3
        w_s = 0


        P_dot = B**2 *np.pi**2 * radius**6 * (1+ np.sin(a)**2) / (P_old*I*c**3)
#         P_dot = (B / 3.1782086e+19)**2 * (1+ np.sin(a)**2) / P_old
        P_dot_mdb = P_dot
#         P_dot = P_dot + Pdot_GW
    #     print(P,P_dot)
        P = P + P_dot * ddt *3.154e+7
        Pn = P
        wn = ( 2 * np.pi / P ) * 3.154e+7
        

        m =  mmass * 1.989e+33  ##g
        r =  rradius * 6.957e+10     ##cm
        I_old = (0.4)*(m*r**2)      ##g cm^2
    #     wn = ww + w_dot* ddt *3.154e+7
        delKE_mdb = 0.5 * (I_old*wn**2 - I_old*ww**2)
    #     print(delKE_mdb/(1e7*3.154e+7)**2)

        flag = 3
        P_dot_final = P_dot_mdb


    #     print(P_dot_acc, P_dot_prop, P_dot_mdb)
    #     print(delKE_acc, delKE_prop, delKE_mdb)

#     P_dot = P_dot_acc+ P_dot_prop+ P_dot_mdb+ P_dot_GW
    if abs(P_dot_mdb) > abs(P_dot_prop):
        flag = 3
    if abs(P_dot_mdb) < abs(P_dot_prop):
        flag = 2
    if abs(P_dot_GW) > abs(P_dot_mdb) and abs(P_dot_GW) > abs(P_dot_prop):
        flag = 1
#     P = P_old + P_dot * ddt *3.154e+7
    w = ( 2 * np.pi / P ) * 3.154e+7
    P_dot = (P - P_old)/(ddt*3.154e+7)
#     print(w, P)
    return P, w, w_s, P_dot, flag, alp, P_dot_mdb, delKE_acc, delKE_prop, delKE_mdb
        

def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
class Distributions:
    def process(self, i, B, a, eta, eta_g):
        G = 1.3218607e+26               # km**3 * MSun**(-1) * yr**(-2)
        G_ = 1.3218607e+26/(695700)**3               # km**3 * MSun**(-1) * yr**(-2)
        g = 6.67408e-11 |(units.m**3 * units.kg**(-1) * units.s**(-2))
        G = g.value_in(units.RSun**3 * units.MSun**(-1) * units.yr**(-2))
        # c = 9.4605284e12                # km/yr
        RSun = 695700   ## km
        c = 2.998e10 ## cm/s
        cc = 2.998e10 |units.cm / units.s  ## cm/s
        
        delKE_acc = 0
        delKE_prop = 0
        delKE_mdb = 0
        
#         i = aic_indices[j]
        
        t = np.arange(0, t_end, dt)
        time_i = ehists[i][:,0]
        self.birthtimes.append(time_i[0])
        types_i = ehists[i][:,5]
        for k in range(len(types_i)):
            if types_i[k] == 13 and types_i[k-1] == 12:
                aic_ind = k
                self.t_aic.append(time_i[aic_ind])
                break

        begining = True 
        j = 0
#         self.pp = []
#         self.ll = []
#         B = ehists[i][-1,14]
#         B = B_list[i]
#         B = random.choice(B_sim_chris)
#         a = a_list[i]
        
        a_d_old = ehists[i][j,2]
#         self.dat[i].append([B, a])
        mic = False
        for t_i in t:
            j_old = j
            j = find_nearest(time_i, t_i/1e6)
            if ehists[i][j,2] == ehists[i][j-1,2]:
                self.mic[int(t_i/dt)]+=1
                mic = True
            if j==j_old:
                mdot = 0
                mloss = 0
            else:
                mdot = ehists[i][j,6]
                mloss = ehists[i][j,10]
            a_d = ehists[i][j,2]
            M1 = ehists[i][j,3]
            M2 = ehists[i][j,7]
            if a_d == a_d_old:
                cc = 9.4605284e12                # km/yr
                adot_grav = - (64/5) * (G**3 * M1*M2*(M1+M2)) / (cc**5 * (a_d_old*RSun)**3)
                a_d = a_d_old + adot_grav/RSun * dt
            if j ==j_old:
                P_orb = None
            else:
                P_orb = 2*np.pi* np.sqrt(a_d**3/(G*(M1+M2)))*525600     ##minutes
            a_d_old = a_d
#             if types_i[j] == 13 and i in aic_indices:
            if types_i[j] == 13:
                w_breakup = np.sqrt(G*ehists[i][j,3]/(ehists[i][j,4]*RSun)**3)
                P_breakup = ( 2 * np.pi /(w_breakup) ) * 3.154e+7    #second
                if accretion == False:
                    mass = ehists[i][j,3] * 1.98e33
                    radius = ehists[i][j,4] * 695700 * 1e5
                    c = 2.998e10 ## cm/s
                    I = (0.4)*(mass*radius**2)      ##g cm^2
                    if begining == True:
#                         P_old = ehists[i][j,12]
#                         P = ehists[i][j,12]
#                         w_ns = ( 2 * np.pi / ( P) ) * 3.154e+7
                        l = 1
                        w_wd = ( 2 * np.pi / (ehists[i][j-l,12]) ) * 3.154e+7  
                        w_ns = ( (ehists[i][j-l,3] / ehists[i][j,3]) * ( ehists[i][j-l,4]/ehists[i][j,4] )**2 ) * w_wd
                        P = ( 2 * np.pi / ( w_ns) ) * 3.154e+7  #seconds
                        if P<0.8e-3:
                            P = ehists[i][j,12]
                            w_ns = ( 2 * np.pi / ( P) ) * 3.154e+7
                            if P<0.8e-3:
                                P = 1e-3
                                w_ns = ( 2 * np.pi / ( P) ) * 3.154e+7
                        self.Pbirth.append(P)
                        alp = np.random.random()
                        mass =  ehists[i][j,3] * 1.989e+33  ##g
                        r =  ehists[i][j,4] * 6.957e+10     ##cm
                        I = (0.4)*(mass*r**2)      ##g cm^2
#                         self.KE0.append(0.5*I*w_ns**2/3.154e+7**2)
                        KE_ = 0.5*I*w_ns**2
                        
                        self.mass.append(ehists[i][j,3])
            
                        self.Pbirth.append(P)
                        alp = np.random.random()
#                         alp = 1e-6
                        P, w, w_s, P_dot, flag, alp, P_dot_mdb1, delKE_acci, delKE_propi, delKE_mdbi = magnetic_braking(P, 
                                                                500e-2/3.154e+7, B, ehists[i][j,3], mdot, ehists[i][j,4], a, True, alp, ehists[i][j-1,12], ehists[i][j-1,3], ehists[i][j-1,4])
                        KE__ = 0.5*I*w**2
                        self.delKE_gw.append((KE__ - KE_)/3.154e+7**2)
                        self.KE0.append(0.5*I*w**2/3.154e+7**2)

                        
                        P, w, w_s, P_dot, flag, alp, P_dot_mdb2, delKE_acci, delKE_propi, delKE_mdbi = magnetic_braking(P, 
                                                                dt, B, ehists[i][j,3], mdot, ehists[i][j,4], a, False, alp)
                        delKE_acc += delKE_acci
                        delKE_prop += delKE_propi
                        delKE_mdb += delKE_mdbi
                        
#                         P_dot_mdb = P_dot_mdb1+P_dot_mdb2
                        P_dot_mdb = P_dot_mdb2
                        begining = False
                        P_old = P 
                        P_dot_old = P_dot
                        begining = False
                    else:
                        P = P_old
                        P, w, w_s, P_dot, flag, alp, P_dot_mdb, delKE_acci, delKE_propi, delKE_mdbi = magnetic_braking(P, 
                                                                                                    dt, B, ehists[i][j,3], mdot, ehists[i][j,4], a, False, alp)
                        delKE_acc += delKE_acci
                        delKE_prop += delKE_propi
                        delKE_mdb += delKE_mdbi
                        P_old = P 
                        P_dot_old = P_dot
                    if P<40e-3 and P_dot > 0 and P_dot_old > 0:
                        self.n[int(t_i/dt)].append( 2 - (P*(P_dot-P_dot_old)/(dt * 3.154e+7))/P_dot**2 )
                    P_dot_old = P_dot

                if accretion == True:
                    if begining == True:
                        l = 1
                        w_wd = ( 2 * np.pi / (ehists[i][j-l,12]) ) * 3.154e+7  
                        w_ns = ( (ehists[i][j-l,3] / ehists[i][j,3]) * ( ehists[i][j-l,4]/ehists[i][j,4] )**2 ) * w_wd
                        P = ( 2 * np.pi / ( w_ns) ) * 3.154e+7  #seconds
                        if P<0.8e-3:
                            P = ehists[i][j,12]
                            w_ns = ( 2 * np.pi / ( P) ) * 3.154e+7
                            if P<0.8e-3:
                                P = np.random.random()*100e-3
                                w_ns = ( 2 * np.pi / ( P) ) * 3.154e+7
                        
#                         P = ehists[i][j,12]
#                         w_ns = ( 2 * np.pi / ( P) ) * 3.154e+7
                        mass =  ehists[i][j,3] * 1.989e+33  ##g
                        r =  ehists[i][j,4] * 6.957e+10     ##cm
                        I = (0.4)*(mass*r**2)      ##g cm^2
#                         self.KE0.append(0.5*I*w_ns**2/3.154e+7**2)
                        KE_ = 0.5*I*w_ns**2
                        
                        self.mass.append(ehists[i][j,3])
            
                        self.Pbirth.append(P)
                        alp = np.random.random()
#                         alp = 1e-6
                        P, w, w_s, P_dot, flag, alp, P_dot_mdb1, delKE_acci, delKE_propi, delKE_mdbi = magnetic_braking(P, 
                                                                500e-2/3.154e+7, B, ehists[i][j,3], mdot, ehists[i][j,4], a, True, alp, ehists[i][j-1,12], ehists[i][j-1,3], ehists[i][j-1,4])
                        KE__ = 0.5*I*w**2
                        self.delKE_gw.append((KE__ - KE_)/3.154e+7**2)
#                         delKE_acc += delKE_acci
#                         delKE_prop += delKE_propi
#                         delKE_mdb += delKE_mdbi
#                         if j==k:
                        self.KE0.append(0.5*I*w**2/3.154e+7**2)
#                         if 0.5*I*w**2/3.154e+7**2 > 1e52:
#                             print(P, mass, r)
                        
                        P, w, w_s, P_dot, flag, alp, P_dot_mdb2, delKE_acci, delKE_propi, delKE_mdbi = magnetic_braking(P, 
                                                                dt, B, ehists[i][j,3], mdot, ehists[i][j,4], a, False, alp)
                        delKE_acc += delKE_acci
                        delKE_prop += delKE_propi
                        delKE_mdb += delKE_mdbi
                        
#                         P_dot_mdb = P_dot_mdb1+P_dot_mdb2
                        P_dot_mdb = P_dot_mdb2
                        begining = False
                        P_old = P 
                        self.aic.append(self.aic[-1]+1)
                    else: 
                        P = P_old
                        P, w, w_s, P_dot, flag, alp, P_dot_mdb, delKE_acci, delKE_propi, delKE_mdbi = magnetic_braking(P, 
                                                                                                    dt, B, ehists[i][j,3], mdot, ehists[i][j,4], a, False, alp)
                        delKE_acc += delKE_acci
                        delKE_prop += delKE_propi
                        delKE_mdb += delKE_mdbi
                        P_old = P 
#                 self.pp.append(P)
                f, alpha, beta = 0.0122, -2.12, 0.82                                   
                L_gamma = 6.8172e35 * f * (P/1e-3)**alpha * (P_dot_mdb/1e-20)**beta 
#                 self.ll.append(L_gamma)
#                 self.dat[i].append([t_i/1e9, P, P_dot_mdb, ehists[i][j,3],
#                                     ehists[i][j,7], ehists[i][j,9], a_d_old])

                        
                        
                        
                        

                
#                 print(delKE_mdb/(dt*3.154e+7)**2)
#                 if P<40e-3 and P_dot > 0:
                if P_dot > 0:
                    mass =  ehists[i][j,3] * 1.989e+33  ##g
                    r =  ehists[i][j,4] * 6.957e+10     ##cm
                    I = (0.4)*(mass*r**2)      ##g cm^2
                    E_dot_mdb = 4 * np.pi**2 * I * P_dot_mdb/P**3     # g cm^2 / s^3 = ergs/s
                    E_dot = 4 * np.pi**2 * I * P_dot/P**3     # g cm^2 / s^3 = ergs/s
#                     ag, bg, dg = 1.2, 0.1, 0.5
#                     aE, bE = 0.18, 2.83
#                     sigma = 0.23
#                     mu = aE * np.log10(E_dot/10**(34.5)) + bE
#                     s = np.random.normal(mu, sigma, int(1e3))
#                     x = np.linspace(min(10**s), max(10**s), int(1e3))
#                     p = (np.sqrt(2*np.pi)*sigma)**-1 * np.exp( -(np.log10(x)-np.log10(10**mu))**2 /(2*sigma**2)  )
#                     p = p/len(p)/np.mean(p)
#                     Ecut = np.random.choice(x,p=p)*0.00160218  ##erg
#                     L_gamma=eta_g * Ecut**ag * B**bg * E_dot**dg
                    f, alpha, beta = 0.0122, -2.12, 0.82                                    ##slot-gap two-pole caustic (TPC) Gonthier 2018
                    L_gamma = 6.8172e35 * f * (P/1e-3)**alpha * (P_dot_mdb/1e-20)**beta 
                    self.dNdL_gamma[int(t_i/dt)].append( L_gamma )
                    self.Power[int(t_i/dt)].append( 1/P**2 )
                    self.P_dist[int(t_i/dt)].append( P )
                    self.P_dot_dist[int(t_i/dt)].append( P_dot )
                    self.P_dot_mdb_dist[int(t_i/dt)].append( P_dot_mdb )
                    self.B_dist[int(t_i/dt)].append( B ) 
#                     if L_gamma>1e33:
#                         self.detectable[int(t_i/dt)].append( True )
#                     else:
#                         self.detectable[int(t_i/dt)].append( False )
                    self.Edot[int(t_i/dt)].append(E_dot)
                    self.Edot_mdb[int(t_i/dt)].append(E_dot_mdb)
                    self.k_donor[int(t_i/dt)].append( ehists[i][j,9] )
#                     self.brake_flags[int(t_i/dt)].append( flag )
#                     T = 2*np.pi * np.sqrt(ehists[i][-1,2]**3/(G_*(ehists[i][j,3]+ehists[i][j,7])))
#                     if T*365<1:
#                         if ehists[i][j,7]>=0.1:
#                             self.redbacks[int(t_i/dt)] += 1
#                         if 0.1>ehists[i][j,7]>=0.01:
#                             self.blackwidows[int(t_i/dt)] += 1
                    if mic == True:
                        self.L_mic[int(t_i/dt)].append( L_gamma )
            
                    

            if ehists[i][j,5] in [10,11,12,13] or ehists[i][j,9] in [10,11,12,13]:  ## if primary is an accreting WD or NS
                if mdot>0:   ##primary accreting
#                     eta = 1.0  ##η_bol ≈ 0.55
                    R_sch = 2 * G * (ehists[i][j,3]) / c**2
                    xi_ = 0.5 * R_sch / (ehists[i][j,4])
                    L = eta * xi_ * abs(mdot) * c**2 
                    L = L * 2e33 * (6.957e+10)**2 / (3.154e+7)**3 
#                     eta = 0.55 * 0.01  ##η_bol ≈ 0.55
#                     L = eta * G * ehists[i][j,3] * abs(ehists[i][j,6]) / (ehists[i][j,4])
#                     L = L * 1.989e33 * (6.957e+10)**2 / (3.154e+7)**3 

                elif mloss>0: ##secondary accreting
#                     eta = 1.0   ##η_bol ≈ 0.55
                    R_sch = 2 * G * (ehists[i][j,7]) / c**2
                    xi_ = 0.5 * R_sch / (ehists[i][j,8])
                    L = eta * xi_ * abs(mloss) * c**2 
                    L = L * 2e33 * (6.957e+10)**2 / (3.154e+7)**3 
#                     eta = 0.55 * 0.01  ##η_bol ≈ 0.55
#                     L = eta * G * ehists[i][j,7] * abs(ehists[i][j,10]) / (ehists[i][j,8])
#                     L = L * 1.989e33 * (6.957e+10)**2 / (3.154e+7)**3 
                else:
                    L = 0
                if L > 0:
                    self.dNdL_x[int(t_i/dt)].append( L ) 
                    if ehists[i][j,5] in [10,11,12]:
                        self.dNdL_xwd[int(t_i/dt)].append( L ) 
                    if ehists[i][j,5] == 13:
                        self.dNdL_xns[int(t_i/dt)].append( L )
                    if L >= 1e36:
                        self.Lx_count[int(t_i/dt)] += 1
                    self.Lx_count_all[int(t_i/dt)] += 1
                    self.P_orb[int(t_i/dt)].append( P_orb )
        
        if types_i[j] == 13:            
            self.delKE_acc.append(delKE_acc/3.154e+7**2)
            self.delKE_prop.append(delKE_prop/3.154e+7**2)
            self.delKE_mdb.append(delKE_mdb/3.154e+7**2)
            mass =  ehists[i][j,3] * 1.989e+33  ##g
            r =  ehists[i][j,4] * 6.957e+10     ##cm
            I = (0.4)*(mass*r**2)      ##g cm^2
            self.KE138.append(0.5*I*w**2/3.154e+7**2)
            if ehists[i][j,7] <0.001:
                self.isolated.append(1)
            else:
                self.isolated.append(0)
                    
                    
    def __init__(self, indices):
        t = np.arange(0, t_end, dt)
        self.dNdL_gamma = [[] for _ in range(len(t))]
        self.Edot = [[] for _ in range(len(t))]
        self.Edot_mdb = [[] for _ in range(len(t))]
        self.L_g_noaic = [[] for _ in range(len(t))]
        self.P_dist = [[] for _ in range(len(t))]
        self.P_dot_dist = [[] for _ in range(len(t))]
        self.P_dot_mdb_dist = [[] for _ in range(len(t))]
        self.Power = [[] for _ in range(len(t))]
        self.B_dist = [[] for _ in range(len(t))]
        self.age = [[] for _ in range(len(t))]
        self.dNdL_x = [[] for _ in range(len(t))]
        self.dNdL_xwd = [[] for _ in range(len(t))]
        self.dNdL_xns = [[] for _ in range(len(t))]
        self.birthtimes = []
        self.detectable = [[] for _ in range(len(t))]
        self.Lx_count = [0 for _ in range(len(t))]
        self.Lx_count_all = [0 for _ in range(len(t))]
        self.t_aic = []
        self.n = [[] for _ in range(len(t))]
        self.Pbirth = []
        self.brake_flags = [[] for _ in range(len(t))]
        self.redbacks = [0 for _ in range(len(t))]
        self.blackwidows = [0 for _ in range(len(t))]
        self.KE0 = []
        self.KE138 = []
        self.delKE_acc = []
        self.delKE_prop = []
        self.delKE_mdb = []
        self.delKE_gw = []
        self.mass = []
        self.finalmass = []
        self.isolated = []
        self.pp = []
        self.ll = []
        self.aic = [0]
        self.mic = [0 for _ in range(len(t))]
        self.dat = [[] for _ in range(len(ehists))]
        self.P_orb = [[] for _ in range(len(t))]
        self.k_donor = [[] for _ in range(len(t))]
        self.L_mic = [[] for _ in range(len(t))]
        
        B_list = np.random.choice(B_sam, len(ehists))
        a_list = np.random.choice(a_sam, len(ehists))
        eta = np.random.choice(etalist, len(ehists))
        eta_g = np.random.choice(eta_sam, len(ehists))
        
        

#         with tqdm(total=len(ehists)) as pbar:
#             for j in range(len(ehists)):
        with tqdm(total=len(indices)) as pbar:
            for j in indices:
                self.process(j, B_list[j], a_list[j], eta[j], eta_g[j])
                pbar.update()
        print("Done!")
#             ncores = 2
#             args = [(ehists, aic_indices[i]) for i in range(len(aic_indices))]
#             print(args)
#             with mp.Pool(processes=ncores) as pool:

#                 for i in enumerate(pool.starmap(self.process, 
#                                             args )):    
#                     pbar.update()

                    

def call_distributions(indices):
    return Distributions(indices)


al = np.linspace(0, np.pi/2, int(1e3))
alpdf = 0.5*np.sin(al)

n_ = 1e4
a_sam = []
for i in range(len(al)):
    a_sam += [al[i]]*int(n_*alpdf[i])

etalist = np.arange(0.1, 0.3, 0.001)
## eta_L
mu = 12    ##log10x_med = mu
sigma = 0.52
s = np.random.normal(mu, sigma, int(1e6))
x = np.linspace(min(10**s), max(10**s), int(4e3))
p = (np.sqrt(2*np.pi)*sigma)**-1 * np.exp( -(np.log10(x)-np.log10(10**mu))**2 /(2*sigma**2)  )
eta_sam = []
for i in range(len(x)):
    eta_sam += [x[i]]*int(len(x)*p[i])

## B
mu = 8.21
sigma = 0.21
s = np.random.normal(mu, sigma, int(1e6))
B_sim_chris = 10**s
x = np.linspace(min(B_sim_chris), max(B_sim_chris), int(4e3))
p = (np.sqrt(2*np.pi)*sigma)**-1 * np.exp( -(np.log10(x)-np.log10(10**mu))**2 /(2*sigma**2)  )
B_sam = []
for i in range(len(x)):
    B_sam += [x[i]]*int(len(x)*p[i])


ehists = read_eval("OutputFiles")
aic_indices = aic_index(ehists)


dt = 1e8
t_end = 14e9
accretion = True
# accretion = False
# dist_disk = call_distributions(ehists_disk, accretion, dt)
xi = 0.5
ehists = copy.copy(ehists)
print("Getting the luminosity data....")
dist1 = call_distributions(aic_indices)


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

save_object(dist1, 'dist1.pkl')

# with open('dist1.pkl', 'rb') as input:
#     dist1 = pickle.load(input)
