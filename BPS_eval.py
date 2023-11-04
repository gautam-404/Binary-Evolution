import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import shutil
from amuse.lab import units
import numpy as np
from tqdm import tqdm
from amuse.units import units, constants
import numpy as np
import amuse

# Suppress numpy warnings
np.seterr(divide='ignore', invalid='ignore')

def save_evo_history(filename, index, output_dir):
    """
    Reads an evolution history file and saves it as a numpy file.
    
    Parameters:
    filename (str): The path to the evolution history file.
    index (int): The index of the file.
    output_dir (str): The directory where the numpy file will be saved.
    """
    evo_history = np.loadtxt(filename)
    np.save(os.path.join(output_dir, f"evo_histories{index}"), evo_history)

def save_ehist(filename, index, output_dir):
    """
    Reads an evolution history file and saves it as a numpy file.
    
    Parameters:
    filename (str): The path to the evolution history file.
    index (int): The index of the file.
    output_dir (str): The directory where the numpy file will be saved.
    """
    ehist = np.loadtxt(filename)
    np.save(os.path.join(output_dir, f"ehists{index}"), ehist)

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
    print(len(aic_indices)) 
    return aic_indices

def read_eval(readdir):
    filenames = glob.glob(os.path.join(readdir, 'EvoHist*'))
    outdir = "EHISTS"
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    else:
        shutil.rmtree(outdir)
        os.makedirs(outdir)
    
    length = len(filenames)
    ncores = mp.cpu_count() if mp.cpu_count() <= length else length

    if ncores == 1:
        save_ehist(filenames[0], 0, outdir)
    else:
        with mp.Pool(ncores) as pool:
            iterable = list(zip(filenames, range(length), [outdir]*length))
            for _ in tqdm(pool.starmap(save_ehist, iterable), total=length):
                pass
    
    ehists = []
    ehist_files = glob.glob(os.path.join(outdir, 'ehists*'))
    for file in ehist_files:
        ehist = np.load(file)
        if ehist.ndim == 1:
            ehist = np.array([ehist])
        ehists.append(ehist)
    shutil.rmtree(outdir)
    return ehists


def magnetic_braking(self, p, dt, B, M, mdot, R, inclination, is_aic, alpha, period_wd, old_mass, old_radius):
    """
    Calculates the magnetic braking effect on a binary system.
    
    Parameters:
    period (float): The initial period of the binary system.
    delta_time (float): The time step for the simulation.
    magnetic_field (float): The magnetic field strength.
    mass (float): The mass of the star.
    mass_loss_rate (float): The mass loss rate of the star.
    radius (float): The radius of the star.
    inclination (float): The inclination angle of the binary system.
    is_aic (bool): Flag indicating if the system is undergoing Accretion Induced Collapse.
    alpha (float): A parameter used in the calculation.
    period_wd (float, optional): The period of the white dwarf. Defaults to 0.
    old_mass (float, optional): The previous mass of the star. Defaults to 1.44.
    old_radius (float, optional): The previous radius of the star. Defaults to 0.0145.
    
    Returns:
    tuple: A tuple containing the updated period, angular velocity, spin parameter, period derivative, flag, alpha, period derivative due to magnetic dipole braking, change in kinetic energy due to accretion, change in kinetic energy due to propeller effect, and change in kinetic energy due to magnetic dipole braking.
    """
    # Initialize variables and constants
    flag = None
    p_old = p.copy()  # seconds
    Omega = (2 * np.pi / p_old)  # rad/s
    delta_energy_acc, delta_energy_prop, delta_energy_mdb = 0, 0, 0
    period_dot_final = 0
    period_dot_acc, period_dot_prop, period_dot_mdb, period_dot_GW = 0, 0, 0, 0

    # Convert units
    radius_cm = R.as_quantity_in(units.cm)
    old_radius_cm = old_radius.as_quantity_in(units.cm)

    # Initialize angular velocities
    angular_velocity_wd = (2 * np.pi / (period_wd | units.s)) if period_wd else 0
    angular_velocity_old = Omega

    # B = B.value_in(units.T) | (units.cm**(-1/2)) * (units.g**(1/2)) * (units.s**-1)

    # AIC-specific calculations
    if is_aic:
        I = 0.4 * M * radius_cm**2
        I_old = 0.4 * old_mass * old_radius_cm**2
        # Define additional constants and calculate timescales
        temperature = 1e11 | units.K
        # Gravitational wave timescale
        t_g = 47 * (M.value_in(units.MSun) / 1.4)**-1 * (radius_cm / (10 | units.km))**-4 * (p / (1e-3 | units.s))**6 
        # Braking timescale
        t_b = 2.7e11 * (M.value_in(units.MSun) / 1.4) * (radius_cm / (10 | units.km))**-1 * (p / (1e-3 | units.s))**2 * (temperature / (1e9 | units.K))**-6 
        # Spin-down timescale
        t_s = 6.7e7 * (M.value_in(units.MSun)/ 1.4)**(-5/4) * (radius_cm / (10 | units.km))**(23/4) * (p / (1e-3 | units.s))**2 * (temperature / (1e9 | units.K))**2 
        # tau = 1/abs(t_b.as_quantity_in(1/units.yr) + t_s.as_quantity_in(1/units.yr) - abs(t_g.as_quantity_in(1/units.yr)))
        tau = (1/abs(t_b + t_s - abs(t_g))) | units.yr

        # Calculate angular momentum and its derivative
        angular_momentum_crit = 3 * 1.635e-2 * alpha**2 * M * radius_cm**2 / 2
        angular_momentum_crit_dot = -2 * angular_momentum_crit / tau

        # Update time step and angular velocities
        dt = tau  # Update time step to tau
        Omega_dot = (Omega - angular_velocity_wd) / dt
        Omega += Omega_dot * dt

        # Update angular momentum and alpha
        angular_momentum_dot = (I - I_old) / dt * Omega + I * Omega_dot + angular_momentum_crit_dot
        Omega_dot = (angular_momentum_dot / I) - (mdot * Omega / M) - (3 * 1.635e-2 * alpha**2 * Omega / (I * tau))
        if 0.01 < alpha < 1:
            Omega_dot = (angular_momentum_dot / I) * (1 - (3 * 1.635e-2 * alpha**2 / (2 * 0.261)))**-1 - (mdot * Omega / M)
        Omega += Omega_dot * dt 

        # Update period and alpha
        p = (2 * np.pi / Omega)
        alpha_dot = -alpha * (1 / tau + Omega_dot / (2 * Omega) + mdot / (2 * M))
        alpha += alpha_dot * dt

        # Calculate period derivative due to gravitational waves
        period_dot_GW = (p - p_old) / (dt)
        period_dot_final = period_dot_GW

        print(p)

    # Non-AIC calculations
    else:
        mdot_lim = 0 | (units.MSun / units.yr)
        if abs(mdot) > mdot_lim:
            # Calculations for mass_loss_rate > mdot_lim
            xi = 1  # Propeller parameter

            mu = B * R**3  # Magnetic moment
            I = 0.4 * M * R**2  # Moment of inertia

            r_A = ((mu**4 / (2 * constants.G * M * mdot**2))**(1.0/7.0))  # Alfvén radius
            r_A = r_A.as_quantity_in(units.km)  # Alfvén radius in km
            r_m = xi * r_A  # Magnetospheric radius
            r_c = (constants.G * M / Omega**2)**(1.0/3.0)  # Co-rotation radius

            w_K_r_m = np.sqrt(constants.G * M / r_m**3)  # Keplerian angular velocity at r_m
            w_K_r_m = w_K_r_m.as_quantity_in(1 / units.yr)
            w_s = Omega / w_K_r_m  # Spin parameter

            if r_m.value_in(units.km) < r_c.value_in(units.km):
                Omega_dot = mdot * np.sqrt(constants.G * M * r_m) / I
                Omega += Omega_dot * dt
                p = (2 * np.pi / Omega)
                period_dot_acc = (p - p_old) / dt
                period_dot_final = period_dot_acc
                flag = 1
            elif r_m.value_in(units.km) >= r_c.value_in(units.km):
                # Calculations for r_m >= r_c
                Omega_dot = - (1 - w_s) * 8.1e-5 * np.sqrt(xi) * (M.value_in(units.MSun) / 1.4)**(3.0/7.0) * (1e45 / I.value_in(units.g * units.cm**2)) * (mu.value_in((units.cm**(-1/2)) * (units.g**(1/2)) * (units.s**-1) * units.m**3) / 1e30)**(2.0/7.0) * (p_old.value_in(units.s) * abs(mdot.value_in(units.MSun / units.yr) / 1e-9)**(3.0/7.0))**2
                Omega_dot = Omega_dot | (units.s**-2)
                period_dot_prop = p**2 * Omega_dot / (2 * np.pi) 
                period_dot_final = period_dot_prop
                p += period_dot_prop * dt
                flag = 2
        else:
            w_s = 0
            # Calculations for mass_loss_rate <= mdot_lim
            # angular_acceleration = B.value_in(units.T/1e4)**2 * np.pi**2 * R.value_in(units.RSun)**6 * (1 + np.sin(inclination)**2) / (p_old.value_in(units.s) * moment_of_inertia.value_in(units.g * units.cm**2) * constants.c.value_in(units.cm / units.s)**3)
            ## magnetic braking
            mu = B * R**3
            I = 0.4 * M * R**2
            Omega = (2 * np.pi / p_old).as_quantity_in(1 / units.s)
            Power_dipole = (2 * mu**2 * Omega**4 * np.sin(inclination)**2 / (3 * constants.c**3))
            Omega_dot = (-Power_dipole / (I * Omega))
            period_dot_mdb = p**2 * Omega_dot / (2 * np.pi)
            p += period_dot_mdb * dt
            flag = 3
            period_dot_final = period_dot_mdb

    # # Determine the dominant braking mechanism
    # if abs(period_dot_mdb) > abs(period_dot_prop):
    #     flag = 3
    # elif abs(period_dot_mdb) < abs(period_dot_prop):
    #     flag = 2
    # elif abs(period_dot_GW) > abs(period_dot_mdb) and abs(period_dot_GW) > abs(period_dot_prop):
    #     flag = 1

    # Update period and calculate period derivative
    pdot = (p - p_old) / (dt.as_quantity_in(units.yr))
    if pdot == np.inf or pdot == -np.inf or pdot == np.nan:
        pdot = 0

    # Return the updated values
    return (p, Omega, w_s, pdot, flag, alpha, period_dot_mdb, delta_energy_acc, delta_energy_prop, delta_energy_mdb)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    

class Distributions:
    def __init__(self, indices, t_end, dt, ehists, B_sam, a_sam, etalist, eta_sam):
        # Time-related quantities
        self.t = np.arange(0, t_end, dt) | units.yr
        self.dt = dt | units.yr
        self.t_end = t_end | units.yr
        
        # Lists initialization (same structure, but now will contain units)
        self.dNdL_gamma = [[] for _ in range(len(self.t))]
        self.Edot = [[] for _ in range(len(self.t))]
        self.Edot_mdb = [[] for _ in range(len(self.t))]
        self.L_g_noaic = [[] for _ in range(len(self.t))]
        self.P_dist = [[] for _ in range(len(self.t))]
        self.P_dot_dist = [[] for _ in range(len(self.t))]
        self.P_dot_mdb_dist = [[] for _ in range(len(self.t))]
        self.Power = [[] for _ in range(len(self.t))]
        self.B_dist = [[] for _ in range(len(self.t))]
        self.age = [[] for _ in range(len(self.t))]
        self.dNdL_x = [[] for _ in range(len(self.t))]
        self.dNdL_xwd = [[] for _ in range(len(self.t))]
        self.dNdL_xns = [[] for _ in range(len(self.t))]
        self.birthtimes = []
        self.detectable = [[] for _ in range(len(self.t))]
        self.Lx_count = [0 for _ in range(len(self.t))]
        self.Lx_count_all = [0 for _ in range(len(self.t))]
        self.t_aic = []
        self.n = [[] for _ in range(len(self.t))]
        self.Pbirth = []
        self.brake_flags = [[] for _ in range(len(self.t))]
        self.redbacks = [0 for _ in range(len(self.t))]
        self.blackwidows = [0 for _ in range(len(self.t))]
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
        self.mic = [0 for _ in range(len(self.t))]
        self.dat = [[] for _ in range(len(ehists))]
        self.P_orb = [[] for _ in range(len(self.t))]
        self.k_donor = [[] for _ in range(len(self.t))]
        self.L_mic = [[] for _ in range(len(self.t))]
        
        # Sampling from input arrays
        self.B_list = np.random.choice(B_sam, len(ehists)) | units.T  # Tesla
        self.a_list = np.random.choice(a_sam, len(ehists)) | units.RSun  # Solar radius
        self.eta = np.random.choice(etalist, len(ehists))  # Dimensionless
        self.eta_g = np.random.choice(eta_sam, len(ehists))  # Dimensionless

        ## Constants ##
        self.G = constants.G
        self.c = constants.c
        self.Msun = 1 | units.MSun
        self.Rsun = 1 | units.RSun
        self.yr = 1 | units.yr
        self.s = 1 | units.s
        self.g = 1 | units.g
        self.cm = 1 | units.cm
        self.km = 1 | units.km
        
        # Process the distributions
        self.process(indices, ehists)
        
        print("Done!")
        
    def process(self, indices, ehists):
        for j in tqdm(indices):
            if isinstance(ehists[j], np.ndarray) or isinstance(ehists[j], list):
                self.calculate_distributions(j, ehists)
                
    def calculate_distributions(self, j, ehists):
        if ehists[j][5] == 13:
            self.calculate_ke_and_isolated(j, ehists)
            self.calculate_gamma_ray_luminosity(j, ehists)
        if ehists[j][5] in [10, 11, 12, 13] or ehists[j][9] in [10, 11, 12, 13]:
            if ehists[j][6] > 0:
                self.calculate_xray_luminosity(j, ehists[j][3], ehists[j][4], ehists[j][6], ehists)
            elif ehists[j][10] > 0:
                self.calculate_xray_luminosity(j, ehists[j][7], ehists[j][8], ehists[j][10], ehists)
                
    def calculate_ke_and_isolated(self, j, ehists):
        P = ehists[j][12] | units.s
        mdot = ehists[j][6] | units.MSun / units.yr
        w_ns = (2 * np.pi / P).as_quantity_in(1/units.s)
        if P < (0.8e-3 | units.s):
            P = (np.random.random() * 100e-3) | units.s
            w_ns = (2 * np.pi / P).as_quantity_in(1/units.s)
        mass = ehists[j][3] * self.Msun
        r = ehists[j][4] * self.Rsun
        I = (0.4) * (mass * r ** 2)
        KE_ = 0.5 * I * w_ns ** 2
        self.mass.append(ehists[j][3])
        self.Pbirth.append(P)
        alp = np.random.random()
        P, w, w_s, P_dot, flag, alp, P_dot_mdb1, delKE_acci, delKE_propi, delKE_mdbi = magnetic_braking(
            P.value_in(units.s), (500e-2 | units.yr).value_in(units.s), self.B_list[j], ehists[j][3], mdot.value_in(units.MSun / units.yr), ehists[j][4], self.a_list[j], True, alp,
            ehists[j - 1][12], ehists[j - 1][3], ehists[j - 1][4])
        KE__ = 0.5 * I * w ** 2
        self.delKE_gw.append(((KE__ - KE_) / (1 | units.yr)**2).value_in(units.J))
        self.KE0.append((0.5 * I * w ** 2 / (1 | units.yr)**2).value_in(units.J))
        P, w, w_s, P_dot, flag, alp, P_dot_mdb2, delKE_acci, delKE_propi, delKE_mdbi = magnetic_braking(
            P, self.dt.value_in(units.s), self.B_list[j], ehists[j][3], mdot.value_in(units.MSun / units.yr), ehists[j][4], self.a_list[j], False, alp)
        delKE_acc = delKE_acci
        delKE_prop = delKE_propi
        delKE_mdb = delKE_mdbi
        P_dot_mdb = P_dot_mdb2
        begining = False
        P_old = P
        self.aic.append(self.aic[-1] + 1)
        while P > (1e-3 | units.s) and P_dot > 0:
            P, w, w_s, P_dot, flag, alp, P_dot_mdb, delKE_acci, delKE_propi, delKE_mdbi = magnetic_braking(
                P.value_in(units.s), self.dt.value_in(units.s), self.B_list[j], ehists[j][3], mdot.value_in(units.MSun / units.yr), ehists[j][4], self.a_list[j], False, alp)
            delKE_acc += delKE_acci
            delKE_prop += delKE_propi
            delKE_mdb += delKE_mdbi
            P_old = P
        self.delKE_acc.append((delKE_acc / (1 | units.yr)**2).value_in(units.J))
        self.delKE_prop.append((delKE_prop / (1 | units.yr)**2).value_in(units.J))
        self.delKE_mdb.append((delKE_mdb / (1 | units.yr)**2).value_in(units.J))
        self.finalmass.append(ehists[j][3])
        mass = ehists[j][3] * self.Msun
        r = ehists[j][4] * self.Rsun
        I = (0.4) * (mass * r ** 2)
        self.KE138.append((0.5 * I * w ** 2 / (1 | units.yr)**2).value_in(units.J))
        if ehists[j][7] < (0.001 | units.MSun):
            self.isolated.append(1)
        else:
            self.isolated.append(0)
            
    def calculate_gamma_ray_luminosity(self, j, ehists):
        P_dot_mdb = self.calculate_p_dot_mdb(j, ehists) | (units.s/units.s)
        t_i = ehists[j][0] | units.yr
        dt = self.dt
        mic = self.mic[j]
        f, alpha, beta = 0.0122, -2.12, 0.82
        L_gamma = (6.8172e35 | units.erg/units.s) * f * ((ehists[j][12] | units.s) / (1e-3 | units.s)) ** alpha * (P_dot_mdb / (1e-20 | units.s/units.s)) ** beta
        self.dNdL_gamma[int((t_i/dt).value_in(units.yr))].append(L_gamma)
        self.Power[int((t_i/dt).value_in(units.yr))].append((1 / (ehists[j][12] | units.s)**2).value_in((units.s)**(-2)))
        self.P_dist[int((t_i/dt).value_in(units.yr))].append(ehists[j][12] | units.s)
        self.P_dot_dist[int((t_i/dt).value_in(units.yr))].append(self.calculate_p_dot(j, ehists) | (units.s/units.s))
        self.P_dot_mdb_dist[int((t_i/dt).value_in(units.yr))].append(P_dot_mdb)
        self.B_dist[int((t_i/dt).value_in(units.yr))].append(self.B_list[j] | units.T)
        self.Edot[int((t_i/dt).value_in(units.yr))].append(self.calculate_e_dot(j, ehists))
        self.Edot_mdb[int((t_i/dt).value_in(units.yr))].append(self.calculate_e_dot_mdb(j, ehists))
        self.k_donor[int((t_i/dt).value_in(units.yr))].append(ehists[j][9])
        if mic:
            self.L_mic[int((t_i/dt).value_in(units.yr))].append(L_gamma)
            
    def calculate_p_dot_mdb(self, j, ehists):
        P = ehists[j][12] | units.s
        mdot = ehists[j][6] | (units.MSun / units.yr)
        w_ns = (2 * np.pi / P).as_quantity_in(1/units.s)
        if P < (0.8e-3 | units.s):
            P = (np.random.random() * 100e-3) | units.s
            w_ns = (2 * np.pi / P).as_quantity_in(1/units.s)
        mass = ehists[j][3] | self.Msun
        r = ehists[j][4] | self.Rsun
        I = 0.4 * mass * r ** 2
        KE_ = 0.5 * I * w_ns ** 2
        alp = np.random.random()
        P, w, w_s, P_dot, flag, alp, P_dot_mdb1, delKE_acci, delKE_propi, delKE_mdbi = magnetic_braking(
            P, (500e-2 / constants.year).as_quantity_in(units.s), self.B_list[j] | units.T, mass, mdot, r, self.a_list[j] | units.RSun, True, alp,
            ehists[j - 1][12] | units.s, ehists[j - 1][3] | self.Msun, ehists[j - 1][4] | self.Rsun)
        P, w, w_s, P_dot, flag, alp, P_dot_mdb2, delKE_acci, delKE_propi, delKE_mdbi = magnetic_braking(
            P, self.dt | units.s, self.B_list[j] | units.T, mass, mdot, r, self.a_list[j] | units.RSun, False, alp)
        P_dot_mdb = P_dot_mdb2
        while P > (1e-3 | units.s) and P_dot > 0:
            P, w, w_s, P_dot, flag, alp, P_dot_mdb, delKE_acci, delKE_propi, delKE_mdbi = magnetic_braking(
                P, self.dt | units.s, self.B_list[j] | units.T, mass, mdot, r, self.a_list[j] | units.RSun, False, alp)
        return P_dot_mdb
    
    def calculate_p_dot(self, j, ehists):
        P = ehists[j][12] | units.s
        mdot = ehists[j][6] | (units.MSun / units.yr)
        w_ns = (2 * np.pi / P).as_quantity_in(1/units.s)
        if P < (0.8e-3 | units.s):
            P = (np.random.random() * 100e-3) | units.s
            w_ns = (2 * np.pi / P).as_quantity_in(1/units.s)
        mass = ehists[j][3] | self.Msun
        r = ehists[j][4] | self.Rsun
        I = 0.4 * mass * r ** 2
        KE_ = 0.5 * I * w_ns ** 2
        alp = np.random.random()
        P, w, w_s, P_dot, flag, alp, P_dot_mdb1, delKE_acci, delKE_propi, delKE_mdbi = magnetic_braking(
            P, (500e-2 / constants.year).as_quantity_in(units.s), self.B_list[j] | units.T, mass, mdot, r, self.a_list[j] | units.RSun, True, alp,
            ehists[j - 1][12] | units.s, ehists[j - 1][3] | self.Msun, ehists[j - 1][4] | self.Rsun)
        P, w, w_s, P_dot, flag, alp, P_dot_mdb2, delKE_acci, delKE_propi, delKE_mdbi = magnetic_braking(
            P, self.dt | units.s, self.B_list[j] | units.T, mass, mdot, r, self.a_list[j] | units.RSun, False, alp)
        while P > (1e-3 | units.s) and P_dot > 0:
            P, w, w_s, P_dot, flag, alp, P_dot_mdb, delKE_acci, delKE_propi, delKE_mdbi = magnetic_braking(
                P, self.dt | units.s, self.B_list[j] | units.T, mass, mdot, r, self.a_list[j] | units.RSun, False, alp)
        return P_dot
    
    def calculate_e_dot(self, j, ehists):
        P = ehists[j][12] | units.s
        mass = ehists[j][3] | self.Msun
        r = ehists[j][4] | self.Rsun
        I = 0.4 * mass * r ** 2
        P_dot = self.calculate_p_dot(j, ehists) | (units.s/units.s)
        return (4 * np.pi ** 2 * I * P_dot / P ** 3).as_quantity_in(units.erg/units.s)

    def calculate_e_dot_mdb(self, j, ehists):
        P = ehists[j][12] | units.s
        mass = ehists[j][3] | self.Msun
        r = ehists[j][4] | self.Rsun
        I = 0.4 * mass * r ** 2
        P_dot_mdb = self.calculate_p_dot_mdb(j, ehists) | (units.s/units.s)
        return (4 * np.pi ** 2 * I * P_dot_mdb / P ** 3).as_quantity_in(units.erg/units.s)


    def calculate_xray_luminosity(self, j, m, r, mdot, ehists):
        R_sch = (2 * self.G * (m | self.Msun) / self.c ** 2).as_quantity_in(units.km)
        xi_ = 0.5 * R_sch / (r | units.RSun)
        t_i = ehists[j][0] | units.yr
        L = self.eta * xi_ * abs(mdot | (units.MSun/units.yr)) * self.c ** 2
        L = (L * 2e33 * (6.957e+10 | units.cm) ** 2 / (3.154e+7 | units.s) ** 3).as_quantity_in(units.erg/units.s)
        self.dNdL_x[int(t_i / self.dt)].append(L)
        if ehists[j][5] in [10, 11, 12]:
            self.dNdL_xwd[int(t_i / self.dt)].append(L)
        if ehists[j][5] == 13:
            self.dNdL_xns[int(t_i / self.dt)].append(L)
        if L >= 1e36:
            self.Lx_count[int(t_i / self.dt)] += 1
        self.Lx_count_all[int(t_i / self.dt)] += 1
        self.P_orb[int(t_i / self.dt)].append(self.calculate_p_orb(j, ehists))
        
    def calculate_p_orb(self, j, ehists):
        return (2 * np.pi * np.sqrt((ehists[j][-1][2] | units.RSun) ** 3 / (self.G * (ehists[j][3] + ehists[j][7]) | self.Msun))).as_quantity_in(units.s)
    


if __name__ == "__main__":
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

    print("Reading output files...")
    ehists = read_eval("./OutputFiles")
    aic_indices = aic_index(ehists)


    dt = 1e8
    t_end = 14e9
    accretion = True
    # accretion = False
    # dist_disk = call_distributions(ehists_disk, accretion, dt)
    xi = 0.5
    ehists = copy.copy(ehists)
    print("Getting the luminosity data....")
    # dist1 = call_distributions(aic_indices)
    dist1 = Distributions(aic_indices)

    print(dist1.dNdL_gamma[0])
    

    with open('dist1.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(dist1, output, pickle.HIGHEST_PROTOCOL)

