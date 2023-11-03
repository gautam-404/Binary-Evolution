# ## Bulge
# def f_bulge(z):
#     A = -2.62e-2
#     B = 0.384
#     C = -8.42e-2
#     D = 3.254
#     return A*z**2 + B*z + C, D

# ## disk
# def f_disk(z):
#     A = -4.06e-2
#     B = 0.331
#     C = 0.338
#     D = 0.771
#     return A*z**2 + B*z + C, D


# def z(t, t_end):
#     t0 = t_end
#     z_ = np.sqrt((28e9 - t)/t) -1
#     return z_


# def sfh(b_d, dt, t_end):
#     ## SFR
#     t = np.arange(0, t_end, dt)
#     sfh = []
#     for time in t: 
#         if b_d == "Bulge":
#             ft, D = f_bulge(z(time, t_end))
#         elif b_d == "Disk":
#             ft, D = f_disk(z(time, t_end))
#         rate = 10**(max(ft, 0)) - D
#         if rate >= 0:
#             sfh.append(rate)
#         else:
#             sfh.append(0)
#     return np.array(sfh)

import numpy as np
# Suppress runtime warnings
np.seterr(divide='ignore', invalid='ignore')

def sample_birth_times(dt, t_end, M_sim, length, bulge_or_disk):
    """
    Sample birth times for stars based on the star formation history.
    
    Parameters:
    dt (float): Time step for sampling.
    t_end (float): End time for the simulation.
    M_sim (float): Total mass to be simulated.
    length (int): Number of birth times to sample.
    bulge_or_disk (str): "Bulge" or "Disk" indicating where the stars are forming.
    
    Returns:
    numpy.ndarray: Array of sampled birth times.
    """
    print("Sampling birth times...")
    M_bulge_initial = 2e10
    sfr = calculate_star_formation_rate(bulge_or_disk, dt, t_end)
    birth_times = formed_at_time(dt, t_end, M_bulge_initial, M_sim, sfr, length)
    birth_times = np.random.choice(birth_times, size=length, replace=True)
    # np.savez_compressed("tr.npz", birth_times)
    return birth_times

def calculate_star_formation_rate(bulge_or_disk, dt, t_end):
    """
    Calculate the star formation rate based on the galaxy's bulge or disk.
    
    Parameters:
    bulge_or_disk (str): "Bulge" or "Disk" indicating where the stars are forming.
    dt (float): Time step for the star formation rate calculation.
    t_end (float): End time for the simulation.
    
    Returns:
    numpy.ndarray: Array of star formation rates at each time step.
    """
    t = np.arange(0, t_end, dt)
    if bulge_or_disk == "Bulge":
        A, B, C, D = -2.62e-2, 0.384, -8.42e-2, 3.254
    elif bulge_or_disk == "Disk":
        A, B, C, D = -4.06e-2, 0.331, 0.338, 0.771
    else:
        raise ValueError("Invalid value for bulge_or_disk. Must be 'Bulge' or 'Disk'.")
    
    z = np.sqrt((28e9 - t) / t) - 1
    ft = A * z**2 + B * z + C
    rate = 10**(np.maximum(ft, 0)) - D
    rate = np.nan_to_num(rate, nan=0)
    return np.maximum(rate, 0)

def formed_at_time(dt, t_end, M_bulge_initial, M_sim, sfr, l):
    """
    Calculate the number of stars formed at each time step.
    
    Parameters:
    dt (float): Time step for the calculation.
    t_end (float): End time for the simulation.
    M_bulge_initial (float): Initial mass of the bulge.
    M_sim (float): Total mass to be simulated.
    sfr (numpy.ndarray): Array of star formation rates at each time step.
    l (int): Scaling factor for the number of stars.
    
    Returns:
    numpy.ndarray: Array of times at which stars are formed.
    """
    rate = sfr * (M_sim / M_bulge_initial)
    normalized_sfr = (rate / rate.sum()) * l
    t = np.arange(0, t_end, dt)
    birth_times = np.repeat(t, normalized_sfr.astype(int))
    return birth_times

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dt = 1e7
    t_end = 14e9
    M_sim = 1e9
    length = int(1e6)
    bulge_or_disk = "Disk"
    tr = sample_birth_times(dt, t_end, M_sim, length, bulge_or_disk)
    tr = tr/1e9
    plt.hist(tr, bins=1000)
    plt.yscale("log")
    plt.show()