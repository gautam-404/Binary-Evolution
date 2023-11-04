import numpy as np
import pandas as pd
from amuse.rfi.core import CodeInterface
from amuse.lab import units, Particles, BSE
import copy
import math
import os

np.seterr(divide='ignore', invalid='ignore')


class BinaryEvolution:
    def __init__(self, code=None):
        self.stars = Particles(2)
        self.binaries = Particles(1)
        self.code = code if code else BSE(channel_type='sockets')
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

    def init_star(self, i, star):
        # Initialize a star
        self.stars[i].mass = star.mass
        self.stars[i].radius = star.radius
        self.stars[i].spin = star.spin
        self.stars[i].stellar_type = star.stellar_type
        self.stars[i].core_mass = star.core_mass
        self.stars[i].core_radius = star.core_radius
        self.stars[i].epoch = star.epoch
        self.stars[i].initial_mass = star.initial_mass
        self.stars[i].luminosity = star.luminosity

    def stars_copy(self, primary_old, secondary_old):
        self.stars =  Particles(2)
        self.init_star(0, primary_old)
        self.init_star(1, secondary_old)
        self.code.particles.add_particles(self.stars)

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

    def binary_copy(self, current_time, a, e):
        # Copy binary data
        self.binary = self.binaries[0]
        self.binary.semi_major_axis = a
        self.binary.eccentricity = e
        self.binary.age = current_time
        self.binary.child1 = self.stars[0]
        self.binary.child2 = self.stars[1]

        self.code.binaries.add_particles(self.binaries)
        self.from_bse_to_model.copy()
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

    def spin_ns_aic(self, primary_old, primary):
        # Calculate spin for NS AIC
        M_ns = primary.mass.value_in(units.MSun)
        M_wd = primary.initial_mass.value_in(units.MSun)
        R = self.wd_mass_to_radius(M_wd)
        r = primary.radius.value_in(units.RSun)
        w_wd = primary_old.spin.value_in(units.none)
        w_ns = (M_wd / M_ns) * (R / r)**2 * w_wd
        P_ns = (2 * np.pi / w_ns) * 3.154e+7  # seconds
        return P_ns, w_ns

    def wd_mass_to_radius(self, M_wd):
        # Convert WD mass to radius
        R = 0.01 * (M_wd ** (-1/3))
        return R

    def main_evolution(self):
        # Main evolution logic for the binary system
        self.secondary = self.stars[1]
        self.primary = self.stars[0]
        M_secondary_limit = 0.001
        self.dt_orig = self.dt
        n_wtf = 1
        n_bu = 1
        B_orig = self.B
        w_s = 0
        
        a_old = self.binary.semi_major_axis
        e_old = self.binary.eccentricity
        w_old = self.primary.spin.value_in(units.none)
        P_old = (2 * np.pi / w_old) * 3.154e+7  # seconds
        primary_old_type = self.primary.stellar_type.value_in(units.stellar_type)
        secondary_old_type = self.secondary.stellar_type.value_in(units.stellar_type)
        M_old = self.primary.mass.value_in(units.MSun)
        M_old_secondary = self.secondary.mass.value_in(units.MSun)
        primary_old = copy.deepcopy(self.primary)
        secondary_old = copy.deepcopy(self.secondary)

        while self.primary.mass.value_in(units.MSun) > 0 and self.real_time <= 15e9:
            if self.primary.stellar_type.value_in(units.stellar_type) in [10,11,12]:
                self.dt = 1e8 | units.yr
                
            if self.primary.stellar_type.value_in(units.stellar_type) == 12 and self.binary.semi_major_axis.value_in(units.RSun) < 100:
                self.dt = 1e7 | units.yr

            a_old = self.binary.semi_major_axis
            e_old = self.binary.eccentricity
            w_old = self.primary.spin.value_in(units.none)
            P_old = (2 * np.pi / w_old) * 3.154e+7  # seconds
            primary_old_type = self.primary.stellar_type.value_in(units.stellar_type)
            secondary_old_type = self.secondary.stellar_type.value_in(units.stellar_type)
            M_old = self.primary.mass.value_in(units.MSun)
            M_old_secondary = self.secondary.mass.value_in(units.MSun)
            primary_old = copy.deepcopy(self.primary)
            secondary_old = copy.deepcopy(self.secondary)

            self.current_time += self.dt
            self.real_time += self.dt.value_in(units.yr)

            if self.secondary.mass.value_in(units.MSun) >= M_secondary_limit:
                if not self.recycling and primary_old_type == 13:
                    pass
                else:
                    self.code.evolve_model(self.current_time)
                    self.from_bse_to_model.copy()
                    self.from_bse_to_model_binaries.copy()

            M_dot = ((self.primary.mass - (M_old | units.MSun)) / self.dt).value_in(units.MSun / units.yr)  # Mass capture by primary per year
            w = self.primary.spin.value_in(units.none)
            P = (2 * np.pi / w) * 3.154e+7  # seconds
            P_bse = P
            w_bse = w

            if self.primary.stellar_type.value_in(units.stellar_type) > 9 or self.secondary.stellar_type.value_in(units.stellar_type) > 9:
                if self.primary.stellar_type.value_in(units.stellar_type) == 15:
                    break
                elif self.secondary.stellar_type.value_in(units.stellar_type) == 15 and self.primary.stellar_type.value_in(units.stellar_type) != 13:
                    break
                elif self.primary.stellar_type.value_in(units.stellar_type) == 13:
                    self.dt = 1e7 | units.yr
            elif self.primary.stellar_type.value_in(units.stellar_type) == primary_old_type or self.secondary.stellar_type.value_in(units.stellar_type) == secondary_old_type:
                self.dt = self.dt_orig
            if math.isnan(self.primary.radius.value_in(units.RSun)) or math.isnan(self.primary.mass.value_in(units.MSun)) or math.isnan(self.secondary.radius.value_in(units.RSun)) or math.isnan(self.secondary.mass.value_in(units.MSun)):
                break
            if self.real_time % self.dt_orig.value_in(units.yr) != 0 and self.dt == self.dt_orig:
                self.dt = self.dt_orig
                self.current_time = np.ceil(self.current_time.value_in(units.Myr) / self.dt_orig.value_in(units.Myr)) * self.dt_orig.value_in(units.Myr) | units.Myr
                self.real_time = np.ceil(self.real_time / self.dt_orig.value_in(units.yr)) * self.dt_orig.value_in(units.yr)

            if self.MSP == False:
                if (primary_old_type == 12 and self.primary.stellar_type.value_in(units.stellar_type) == 13) or (secondary_old_type == 12 and self.secondary.stellar_type.value_in(units.stellar_type) == 13):
                    break

            w = self.primary.spin.value_in(units.none)
            P = (2 * np.pi / w) * 3.154e+7  # seconds
            P_new = P

            P_dot = (P_new - P_old) / self.dt.value_in(units.s)
            M_dot = (self.primary.mass.value_in(units.MSun) - M_old) / self.dt.value_in(units.yr)  # Mass capture by primary per year
            M_loss = (self.secondary.mass.value_in(units.MSun) - M_old_secondary) / self.dt.value_in(units.yr)  # Mass lost by secondary per year

            if (M_dot != 0 or M_loss != 0):
                self.ehist_arr.append([self.real_time / 1e6, self.current_time.value_in(units.Myr), self.binary.semi_major_axis.value_in(units.RSun) * 2, self.primary.mass.value_in(units.MSun),
                                       self.primary.radius.value_in(units.RSun), self.primary.stellar_type.value_in(units.stellar_type), M_dot,
                                       self.secondary.mass.value_in(units.MSun), self.secondary.radius.value_in(units.RSun),
                                       self.secondary.stellar_type.value_in(units.stellar_type), M_loss, P, P_dot, self.B])

            if (self.primary.stellar_type.value_in(units.stellar_type) != primary_old_type) and (self.primary.stellar_type.value_in(units.stellar_type) == 13 and primary_old_type != 12):
                break  # Only AIC NSs
            if (not 9 < self.primary.stellar_type.value_in(units.stellar_type) < 14) and self.secondary.stellar_type.value_in(units.stellar_type) > 12:
                break

    def evolve_binary(self):
        # Initialize instance variables
        self.real_time = self.t_birth
        self.secondary = self.stars[1]
        self.primary = self.stars[0]
        self.current_time = 0 | units.yr
        self.dt = 1e6 | units.yr
        self.ehist_arr = []

        self.recycling = True

        self.ehist_arr.append([self.real_time / 1e6, self.current_time.value_in(units.Myr), self.binary.semi_major_axis.value_in(units.RSun) * 2, self.primary.mass.value_in(units.MSun),
                               self.primary.radius.value_in(units.RSun), self.primary.stellar_type.value_in(units.stellar_type), 0,
                               self.secondary.mass.value_in(units.MSun), self.secondary.radius.value_in(units.RSun),
                               self.secondary.stellar_type.value_in(units.stellar_type), 0, 0, 0, self.B])
        self.MSP = False
        # Perform the main evolution
        self.main_evolution()

        primary_old = copy.deepcopy(self.primary)
        secondary_old = copy.deepcopy(self.secondary)
        a_old = self.binary.semi_major_axis

        # Post-evolution checks and updates
        if self.primary.stellar_type.value_in(units.stellar_type) == 13 and self.primary_old.stellar_type.value_in(units.stellar_type) == 12:
            self.AIC = True
            self.t_aic = self.current_time.value_in(units.Myr)
            if self.printing:
                self.print_evolution(self.primary, self.secondary, self.current_time, self.real_time)
        else:
            self.AIC = False
            if self.printing:
                self.print_evolution(self.primary, self.secondary, self.current_time, self.real_time)

        if self.primary.stellar_type.value_in(units.stellar_type) in [10,11,12] or self.secondary.stellar_type.value_in(units.stellar_type) in [10,11,12]:
            self.WD = True
        else:
            self.WD = False

        self.t_aic = self.real_time
        w = self.primary.spin.value_in(units.none)
        self.P_aic = (2 * np.pi / w) * 3.154e+7  # seconds

        if self.AIC or self.WD:
            Pns, wns = self.spin_ns_aic(primary_old, self.primary)
            self.primary.spin = wns | units.none

            M_ns = self.primary.mass.value_in(units.MSun)
            M_wd = self.primary.initial_mass.value_in(units.MSun)
            e = (M_wd - M_ns) / (M_ns + secondary_old.mass.value_in(units.MSun))
            a = a_old * (1 + e)

            self.code.particles.remove_particles(self.stars)
            self.stars_copy(self.primary, self.secondary)
            self.secondary = self.stars[1]
            self.primary = self.stars[0]
            self.code.binaries.remove_particles(self.binaries)
            self.binary_copy(self.current_time, a, e)

            self.main_evolution()

            w = self.primary.spin.value_in(units.none)
            P = (2 * np.pi / w) * 3.154e+7  # seconds

            if self.printing:
                self.print_evolution(self.primary, self.secondary, self.current_time, self.real_time)

        if self.primary.stellar_type.value_in(units.stellar_type) == 13 or self.secondary.stellar_type.value_in(units.stellar_type) == 13:
            self.NS = True

    def parallel_evolution(self, data, i, B, t_birth, ecc, printing, outdir):
        # Parallel evolution of binary systems
        M1_zams, M2_zams, a_zams, e_zams = data
        self.B = B
        self.t_birth = t_birth
        self.ecc = ecc
        self.printing = printing
        self.outdir = outdir
        self.init_binary(M1_zams | units.MSun, M2_zams | units.MSun, a_zams | units.RSun, e_zams)
        ehist_arr = []

        self.evolve_binary()

        if printing:
            print("Evolution done....saving data")
            print(f"t_birth = {t_birth/1e9:.4f} Gyr")
            print(f"M1_zams = {M1_zams:.3f}")
            print(f"M2_zams = {M2_zams:.3f}")
            print(f"a_zams = {a_zams:.3f}")
            print(f"ecc = {ecc:.3f}")
            print(f"current_time = {self.current_time.value_in(units.Myr):.3f} Myr")
            print(f"real_time = {self.real_time/1e9:.3f} Gyr")
            if self.AIC:
                print(f"t_aic = {self.t_aic:.3f}")
                print(f"P_aic = {self.P_aic:.3f}")
                print(f"B = {B:.3f}")
                print(f"AIC = {self.AIC}")
                print(f"WD = {self.WD}")
                print(f"NS = {self.NS}")

        if self.AIC:
            # np.savetxt(os.path.join(outdir, "EvoHist_{}".format(i)), ehist_arr)
            pd.DataFrame(self.ehist_arr, columns=["real_time", "current_time", "a", "M1", "R1", "T1", "Mdot1", "M2", "R2", "T2", "Mdot2", "P", "Pdot", "B"]).to_csv(os.path.join(outdir, f"EvoHist_{i}.csv"), index=False)

        self.code.stop()
        if printing:
            if self.AIC:
                print("AIC detected....saving data.")
                input("Evolution done....continue to next?")
            else:
                print("Evolution done....continuing to next.\n\n")

def evolve(data, i, B, t_birth, ecc, printing, outdir):
    evolver = BinaryEvolution()
    evolver.parallel_evolution(data, i, B, t_birth, ecc, printing, outdir)


if __name__ ==  "__main__":
    evolver = BinaryEvolution()
    data = [7, 3, 100, 0.5]
    i = 0
    B = 1e9
    t_birth = 0
    ecc = 0
    printing = True
    outdir = "./OutputFiles"
    evolver.parallel_evolution(data, i, B, t_birth, ecc, printing, outdir)

