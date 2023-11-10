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

def z_at_time(t):
    """
    Calculate the redshift at a given time.
    
    Parameters:
    t (float): Time in years.
    
    Returns:
    float: Redshift at the given time.
    """
    from astropy.cosmology import Planck18
    z = np.arange(0, 100, 0.01)
    t_ = Planck18.age(z).value
    return np.interp(t/1e9, z, t_)
    ## The Cosmic Time in Terms of the Redshift 2005, Carmeli et al.
    # return np.sqrt(28e9/t - 1) - 1

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
    t_offset = 350e6
    t = np.arange(-t_offset, t_end-t_offset, dt)
    if bulge_or_disk == "Bulge":
        A, B, C, D = -2.62e-2, 0.384, -8.42e-2, 3.254
    elif bulge_or_disk == "Disk":
        A, B, C, D = -4.06e-2, 0.331, 0.338, 0.771
    else:
        raise ValueError("Invalid value for bulge_or_disk. Must be 'Bulge' or 'Disk'.")
    
    z = z_at_time(t)
    ft = A * z**2 + B * z + C
    rate = 10**(np.maximum(ft, 0)) - D
    # rate = np.nan_to_num(rate, nan=0)
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
    bulge_or_disk = "Bulge"
    # tr = sample_birth_times(dt, t_end, M_sim, length, bulge_or_disk)
    tr = calculate_star_formation_rate(bulge_or_disk, dt, t_end)
    tr = tr
    # plt.hist(tr, bins=100)
    t = np.arange(0, t_end, dt)
    plt.plot(t/1e9, tr)
    plt.title(bulge_or_disk)
    plt.yscale("log")
    plt.xlabel("Cosmological Age (Gyr)")
    plt.ylabel("Star Formation Rate (MSun/yr)")
    plt.show()