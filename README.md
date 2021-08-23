# Binary Population Synthesis with BSE

Read the paper for details: [Millisecond Pulsars from Accretion Induced Collapse naturally explain the Galactic Center Gamma-ray Excess](https://arxiv.org/abs/2106.00222)

## Getting started

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
    cd Binary-Evolution
    python BPS.py
    ```
This would create a folder with evolution histories of all systems at "~/OutputFiles". Consecutively, running `python BPS_eval.py` produces the data necessary to see the total gamma-ray luminosity produced by the population as a function of time. 

[Contact](mailto:anujgautam11@gmail.com) for more info.
