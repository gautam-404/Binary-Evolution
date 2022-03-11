# Binary Population Synthesis with BSE

Read the paper for details: [Millisecond Pulsars from Accretion Induced Collapse naturally explain the Galactic Center Gamma-ray Excess](https://arxiv.org/abs/2106.00222)

## Getting started
* *Linux users can install this repository as a package by executing `pip install git+https://github.com/gautam-404/Binary-Evolution.git`*

**For installation from scratch:**

### Prerequisites
* Python
* [AMUSE Framework](https://amuse.readthedocs.io/en/latest/install/howto-install-AMUSE.html)
    <br> 
    ```
    pip install amuse-framework
    ```
    Visit the AMUSE docs link and install all the dependancies
* [AMUSE BSE](https://amuse.readthedocs.io/en/latest/reference/available-codes.html#bse) community code
    <br> 
    ```
    pip install amuse-bse
    ```
<br>
<br>

### Running the code
* Clone the repository
    ```
    git clone https://github.com/gautam-404/Binary-Evolution.git
    ```
* Run 
    ```
    cd Binary-Evolution
    python main.py
    ```
This would create a folder with evolution histories of all binary systems at the path "~/OutputFiles". Consecutively, running `python BPS_eval.py` produces the data necessary to see the total gamma-ray luminosity produced by the population as a function of time. 

Alternatively, a docker image for this project can be pulled using `docker pull algernon11/bps:ubuntu`.

[Contact](mailto:anujgautam11@gmail.com) for more info.
