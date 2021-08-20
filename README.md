# Binary Population Synthesis with BSE

## Introduction
To evolve a model population of binaries up to AIC, we have used the state-of-the-art binary-star evolution ([BSE](http://www.ascl.net/1303.014)) code. BSE takes as input stellar zero age main sequence (ZAMS) masses, $M_{1,ZAMS}$ and $M_{2,ZAMS}$ for the primary and secondary respectively ($M_{1,ZAMS} \ge M_{2,ZAMS}$), and the initial binary separation, $a$. We adopt the following empirically-motivated, “off-the-shelf” parameter choices: i) A binary star fraction of $70\%$; ii) $M_{1,ZAMS}$ masses are drawn from the [Kroupa (2001)](https://ui.adsabs.harvard.edu/abs/2001MNRAS.322..231K/abstract) broken power law initial mass function; iii) $M_{2,ZAMS}$ masses are drawn from a flat mass ratio distribution; iv) Initial eccentricities are set to zero; v) Orbital periods at zero age are drawn from a log-uniform distribution covering $10$ to $10^4$ days; vi) Common envelope efficiency factor is set to $\alpha_{CE} = 1$; vii) The binding energy parameter $\lambda_b$ is calculated using the Cambridge STARS code; and viii) Magnetic fields are assigned to each nascent NS via random sampling from the log-normal distribution of. Our synthetic population consists of $\approx 6.3 \times 10^7$ binary systems which, given these parameters, has a host stellar population of $2.0\times 10^9 M_\odot$ total ZAMS mass. The binaries are numerically evolved up to 14 Gyrs generating a total of $\approx9.1\times10^3$ AIC events, which start when the population is only $\approx0.2$ Gyr old.

Read the paper for more details: [Millisecond Pulsars from Accretion Induced Collapse naturally explain the Galactic Center Gamma-ray Excess](https://arxiv.org/abs/2106.00222)

## Getting started
<br>

### Prerequisites
* Python
* [AMUSE Framework](https://github.com/amusecode/amuse)
    <br> 
    ```
    pip install amuse-framework
    ```
* AMUSE BSE community code
    <br> 
    ```
    pip install amuse-bse
    ```
* Other necessary python ppackages:
    * numpy
    * multiprocessing
    * concurrent.futures
    * itertools
<br>
<br>

### Running the code
* Clone the repository
    ```
    git clone https://github.com/gautam-404/Binary-Evolution.git
    ```
* Run 
    ```
    python BPS.py
    ```