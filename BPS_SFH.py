import numpy as np
import random

def SFH(dt, t_end, M_sim, length, b_d):
    print("Sampling birth times...")
    t = np.arange(0, t_end, dt)
    t_end = 16e9
    dt = 1e6
    M_b = 2e10
    sfr = sfh(b_d, dt, t_end)
    tr_sam = Nformed_at_t(dt, t_end, M_b, M_sim, sfr, int(1e8))
    tr = random.choices(tr_sam,  k=length)
    np.savez_compressed("tr.npz", tr)
    return tr

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


def z(t, t_end):
    t0 = t_end
    z_ = np.sqrt((28e9 - t)/t) -1
    return z_


def sfh(b_d, dt, t_end):
    ## SFR
    t = np.arange(0, t_end, dt)
    sfh = []
    for time in t: 
        if b_d == "Bulge":
            ft, D = f_bulge(z(time, t_end))
        elif b_d == "Disk":
            ft, D = f_disk(z(time, t_end))
        rate = 10**(max(ft, 0)) - D
        if rate >= 0:
            sfh.append(rate)
        else:
            sfh.append(0)
    return np.array(sfh)


def Nformed_at_t(dt, t_end, M_bulge, M_sim, sfr, l):
    rate = sfr*(M_sim/M_bulge)
    sfr = (rate/sum(rate)) * l
    t = np.arange(0, t_end, dt)
    tr = []
    l = 0
    for i in range(len(t)):
        if sfr[i]==0 or sfr[i]==np.nan:
            pass
        else:
            for j in range( int(round(sfr[i])) ):
                tr.append( t[i] )
                l += 1
    return tr     
