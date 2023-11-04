import numpy as np
import pandas as pd
from amuse.lab import units, Particles, BSE
import os
from amuse.units import units, constants

np.seterr(divide='ignore', invalid='ignore')


class BinaryEvolution:
    def __init__(self, code=None):
        self.stars = Particles(2)
        self.binaries = Particles(1)
        self.code = code if code else BSE(channel_type='sockets', number_of_workers=1)
        # self.code = code if code else BSE()
        self.configure_code()
    
    def configure_code(self):
        # Set the parameters for the code
        parameters = self.code.parameters
        parameters.binary_enhanced_mass_loss_parameter = 0
        parameters.common_envelope_efficiency = 1
        parameters.common_envelope_binding_energy_factor = -0.5
        parameters.common_envelope_model_flag = 0
        parameters.Eddington_mass_transfer_limit_factor = 1
        parameters.wind_accretion_factor = 1.5
        parameters.wind_accretion_efficiency = 1
        parameters.Roche_angular_momentum_factor = -1
        parameters.white_dwarf_IFMR_flag = 0
        parameters.white_dwarf_cooling_flag = 1
        parameters.neutron_star_mass_flag = 1
        parameters.fractional_time_step_1 = 0.05
        parameters.fractional_time_step_2 = 0.01
        parameters.fractional_time_step_3 = 0.02

    def init_binary(self, M_primary, M_secondary, a, e):
        # Initialize a binary system
        self.stars[0].mass = M_primary
        self.stars[1].mass = M_secondary

        self.binary = self.binaries[0]
        self.binary.semi_major_axis = a
        self.binary.eccentricity = e
        self.binary.child1 = self.stars[0]
        self.binary.child2 = self.stars[1]

        self.code.particles.add_particles(self.stars)
        self.code.binaries.add_particles(self.binaries)

        self.from_bse_to_model = self.code.particles.new_channel_to(self.stars)
        self.from_bse_to_model.copy()
        self.from_bse_to_model_binaries = self.code.binaries.new_channel_to(self.binaries)
        self.from_bse_to_model_binaries.copy()

    def print_evolution(self, primary, secondary, current_time, real_time):
        # Print the current state of evolution
        print(f"Time = {current_time.value_in(units.Myr)} Myr")
        print(f"Real time = {real_time/1e6} Myr")
        print(f"Semi-major axis = {self.binary.semi_major_axis.value_in(units.RSun)*2} RSun")
        print(f"Primary mass = {primary.mass.value_in(units.MSun)} MSun")
        print(f"Primary radius = {primary.radius.value_in(units.RSun)} RSun")
        print(f"Primary type = {primary.stellar_type.value_in(units.stellar_type)}")
        print(f"Primary spin = {primary.spin.value_in(units.none)}")
        print(f"Secondary mass = {secondary.mass.value_in(units.MSun)} MSun")
        print(f"Secondary radius = {secondary.radius.value_in(units.RSun)} RSun")
        print(f"Secondary type = {secondary.stellar_type.value_in(units.stellar_type)}")
        print(f"Secondary spin = {secondary.spin.value_in(units.none)}")
        print("")

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

        print(p_old)
        print(period_dot_acc, period_dot_GW, period_dot_prop, period_dot_mdb)
        print(flag, p, pdot, w_s)
        # # if pdot!=0:
        exit()

        # Return the updated values
        return (p, Omega, w_s, pdot, flag, alpha, period_dot_mdb)

    def spin_to_period(self, spin):
        # Calculate the period and period derivative
        P = (2 * np.pi / spin.value_in(units.none)) | units.Gyr #seconds
        return P.as_quantity_in(units.s)
    def period_to_spin(self, period):
        # Calculate the period and period derivative
        w = (2 * np.pi / period.value_in(units.s)) | units.none
        return w

    def spin_after_collapse(self, star_old, star):
        # Calculate the angular momentum of the star
        I = star.mass * star.radius**2
        I_old = star_old.mass * star_old.radius**2
        I_ratio = I_old/I
        # Get new spin
        spin = I_ratio * star_old.spin.value_in(units.none) | units.none
        return spin


    def rotation_update(self, primary, secondary, primary_old, secondary_old, P_primary, dt):
        P_primary_old = P_primary.copy()
        primary_old_type = primary_old.stellar_type.value_in(units.stellar_type)
        secondary_old_type = secondary_old.stellar_type.value_in(units.stellar_type)
        mdot_primary = (primary.mass - primary_old.mass) / dt
        mdot_secondary = (secondary.mass - secondary_old.mass) / dt

        if primary.stellar_type.value_in(units.stellar_type) == 13: 
            if primary_old_type != 13:
                P_primary = self.spin_to_period(self.spin_after_collapse(primary_old, primary))
                if secondary_old_type not in [10, 11, 12]:
                    self.B = 1e2 | 1e-4 * (units.cm**(-1/2)) * (units.g**(1/2)) * (units.s**-1)   # Gauss
                elif secondary_old_type in [10, 11]:
                    self.B = 1e5 | 1e-4 * (units.cm**(-1/2)) * (units.g**(1/2)) * (units.s**-1)   # Gauss
                Pdot_primary = (P_primary - P_primary_old)/dt
            
            is_aic = True if primary_old_type == 12 else False
            period_wd = self.spin_to_period(primary_old.spin) if primary_old_type == 12 else 0
            spin_data = self.magnetic_braking(p=P_primary, dt=dt, B=self.B, M=primary.mass, 
                                            mdot=mdot_primary, R=primary.radius, inclination=0, 
                                            is_aic=is_aic, alpha=np.random.random(),
                                            period_wd=period_wd, old_mass=primary_old.mass, 
                                            old_radius=primary_old.radius)
            P_primary = spin_data[0]
            Pdot_primary = spin_data[3]
        else:
            P_primary = self.spin_to_period(primary.spin)
            Pdot_primary = (self.spin_to_period(primary.spin) - self.spin_to_period(primary_old.spin))/dt
        return P_primary, Pdot_primary, mdot_primary, mdot_secondary

    def evolve_binary(self):
        # Initialize instance variables
        secondary = self.stars[1]
        primary = self.stars[0]
        current_time = 0 |units.yr
        dt = self.dt
        AIC = False
        NS = False

        primary_old_type = primary.stellar_type.value_in(units.stellar_type)
        secondary_old_type = secondary.stellar_type.value_in(units.stellar_type)
        primary_old = primary.copy()
        secondary_old = secondary.copy()

        mdot_primary = (primary.mass - primary_old.mass) / dt
        mdot_secondary = (secondary.mass - secondary_old.mass) / dt
        P_primary = self.spin_to_period(primary.spin)
        P_secondary = self.spin_to_period(secondary.spin)
        Pdot_primary = 0
        Pdot_secondary = 0

        self.ehist_arr.append( [self.real_time/1e6, current_time.value_in(units.Myr), self.binary.semi_major_axis.value_in(units.RSun)*2, self.ecc, primary.mass.value_in(units.MSun), mdot_primary.value_in(units.MSun/units.Myr), 
                                primary.radius.value_in(units.RSun), primary.stellar_type.value_in(units.stellar_type), P_primary.value_in(units.s), Pdot_primary, self.B.value_in(1e-4 * (units.cm**(-1/2)) * (units.g**(1/2)) * (units.s**-1)),
                                secondary.mass.value_in(units.MSun), mdot_secondary.value_in(units.MSun/units.Myr), secondary.radius.value_in(units.RSun), secondary.stellar_type.value_in(units.stellar_type)])
        

        while primary.mass.value_in(units.MSun)>0 and self.real_time<=14e9:
            primary_old = primary.copy()
            secondary_old = secondary.copy()
            primary_old_type = primary.stellar_type.value_in(units.stellar_type)
            secondary_old_type = secondary.stellar_type.value_in(units.stellar_type)

            current_time += dt
            self.real_time += dt.value_in(units.yr)

            ## Evolve the binary system
            self.code.evolve_model( current_time )
            self.from_bse_to_model.copy()
            self.from_bse_to_model_binaries.copy()  

            ## Calculations
            P_primary, Pdot_primary, mdot_primary, mdot_secondary = self.rotation_update(primary, secondary, primary_old, secondary_old, P_primary, dt)

                
            ## Ehists
            self.ehist_arr.append( [self.real_time/1e6, current_time.value_in(units.Myr), self.binary.semi_major_axis.value_in(units.RSun)*2, self.ecc, primary.mass.value_in(units.MSun), mdot_primary.value_in(units.MSun/units.Myr), 
                                primary.radius.value_in(units.RSun), primary.stellar_type.value_in(units.stellar_type), P_primary.value_in(units.s), Pdot_primary, self.B.value_in(1e-4 * (units.cm**(-1/2)) * (units.g**(1/2)) * (units.s**-1)),
                                secondary.mass.value_in(units.MSun), mdot_secondary.value_in(units.MSun/units.Myr), secondary.radius.value_in(units.RSun), secondary.stellar_type.value_in(units.stellar_type)])
                                                  
            #### To print the evolution
            if self.printing == True:
                self.print_evolution(primary, secondary, current_time, self.real_time)
            if (primary_old_type == 12 and primary.stellar_type.value_in(units.stellar_type) == 13) or (secondary_old_type == 12 and secondary.stellar_type.value_in(units.stellar_type) == 13):
                AIC = True
            if (primary.stellar_type.value_in(units.stellar_type) == 13 and primary_old_type!=13) or (secondary_old_type != 13 and secondary.stellar_type.value_in(units.stellar_type) == 13):
                NS = True
            if primary.stellar_type.value_in(units.stellar_type) > 13 or secondary.stellar_type.value_in(units.stellar_type) > 13:
                dt = 1e7 |units.yr
        return AIC, NS
    

    def parallel_evolution(self, data, i, dt, B, t_birth, ecc, printing, outdir):
        # Parallel evolution of binary systems
        M1_zams, M2_zams, a_zams, e_zams = data
        # self.B = B | units.T / 10**4
        self.B = B | 1e-4 * (units.cm**(-1/2)) * (units.g**(1/2)) * (units.s**-1)    # Gauss
        self.dt = dt | units.yr
        self.t_birth = t_birth
        self.real_time = t_birth
        self.ecc = ecc
        self.printing = printing
        self.outdir = outdir
        self.init_binary(M1_zams | units.MSun, M2_zams | units.MSun, a_zams | units.RSun, e_zams)
        self.ehist_arr = []

        self.AIC, self.NS = self.evolve_binary()
        self.code.stop()

        if printing:
            print("Evolution done....saving data")
            print(f"t_birth = {t_birth/1e9:.4f} Gyr")
            print(f"M1_zams = {M1_zams:.3f}")
            print(f"M2_zams = {M2_zams:.3f}")
            print(f"a_zams = {a_zams:.3f}")
            print(f"ecc = {ecc:.3f}")
            print(f"current_time = {self.real_time.value_in(units.Myr):.3f} Myr")
            print(f"real_time = {self.real_time/1e9:.3f} Gyr")
            if self.AIC:
                print(f"t_aic = {self.t_aic:.3f}")
                print(f"P_aic = {self.P_aic:.3f}")
                print(f"B = {B:.3f}")
                print(f"AIC = {self.AIC}")
                print(f"WD = {self.WD}")
                print(f"NS = {self.NS}")

        # if self.NS == True:
        df = pd.DataFrame(self.ehist_arr, columns=["real_time", "current_time", "separation", "Ecc", "M1", "Mdot1", "R1",
                            "T1", "P1", "Pdot1", "B", "M2", "Mdot2", "R2", "T2"], dtype=float)
        df['T1'] = df['T1'].astype(int)
        df['T2'] = df['T2'].astype(int)
        df = df.to_csv(os.path.join(outdir, f"EvoHist_{i}.csv"), index=False, float_format='%.5e', sep="\t")

        self.code.stop()
        if printing:
            if self.AIC:
                print("AIC detected....saving data.")
                input("Evolution done....continue to next?")
            else:
                print("Evolution done....continuing to next.\n\n")

def evolve(data, i, dt, B, t_birth, ecc, printing, outdir):
    evolver = BinaryEvolution()
    evolver.parallel_evolution(data, i, dt, B, t_birth, ecc, printing, outdir)


if __name__ ==  "__main__":
    data = [5.17546, 2.04775, 100, 0.5]
    i = 0
    B = 1e9
    t_birth = 0
    dt = 1e5
    ecc = 0
    printing = False
    outdir = "./OutputFiles"
    evolve(data, i, dt, B, t_birth, ecc, printing, outdir)

