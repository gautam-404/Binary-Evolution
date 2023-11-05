import numpy as np
import pandas as pd

def generate_binary_systems(M_tot, tolerance=1e7):
    # Constants and limits
    M1_min, M1_max = 5.0, 10.0
    M2_min, M2_max = 0.5, 5.0
    P_min, P_max = 10.0, 10000.0
    kroupa = -2.3
    G = 1.3218607e+26  # km**3 * MSun**(-1) * yr**(-2)

    # Calculate the number of systems to simulate
    avg_mass = (M1_min + M1_max) / 2 + (M2_min + M2_max) / 2
    n_systems = int(M_tot / avg_mass)

    # Generate primary masses
    u = np.random.uniform(0, 1, n_systems)
    M1 = ((M1_max**(kroupa+1.0) - M1_min**(kroupa+1.0)) * u + M1_min**(kroupa+1.0))**(1.0 / (kroupa+1.0))

    # Generate secondary masses
    M2 = np.random.uniform(M2_min, M2_max, n_systems)

    # Generate periods
    LP_min, LP_max = np.log10(P_min), np.log10(P_max)
    LogPeriod = np.random.uniform(LP_min, LP_max, n_systems)
    Period = (10.0**LogPeriod) / 365  # in years

    # Semi-major axis
    a = (G * (M1 + M2) * (0.5 * Period / np.pi)**2)**(1/3) / 695700  # in RSun

    ## Dataframe
    df = pd.DataFrame({
        'M1': M1,
        'M2': M2,
        'Period': Period,
        'Semi-major axis': a
    })

    # Ensure the total mass is close to M_tot but not exceeded
    total_mass = df['M1'].sum() + df['M2'].sum()
    while total_mass < M_tot - tolerance:
        missing_mass = M_tot - total_mass
        additional_systems = generate_binary_systems(missing_mass)[0]
        df = pd.concat([df, additional_systems])
        df = df[(df['M1'] + df['M2']).cumsum() <= M_tot]
        total_mass = df['M1'].sum() + df['M2'].sum()
    return df, total_mass

# if __name__ == "__main__"
#     M_tot = float(input("Enter the total mass to be simulated (units MSun) \n"))
#     binary_systems, total_mass = generate_binary_systems(M_tot, tolerance=1e7)
#     print(f'Total mass simulated = {total_mass:.4e} MSun')
#     binary_systems.to_csv("Init_data.csv", index=False)
