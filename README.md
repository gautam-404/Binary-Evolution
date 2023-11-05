# Binary Population Synthesis with BSE

Read the paper for details: [Millisecond Pulsars from Accretion Induced Collapse naturally explain the Galactic Center Gamma-ray Excess](https://arxiv.org/abs/2106.00222)

Supplementary reading:
* [Nature Astronomy publication](https://www.nature.com/articles/s41550-022-01658-3)
* [Nature Astronomy blog post](https://astronomycommunity.nature.com/posts/millisecond-pulsars-from-accretion-induced-collapse-naturally-explain-the-galactic-center-gamma-ray-excess)

## Getting started


### Using from scratch:
* Install Python 3
* Visit the [AMUSE docs](https://amuse.readthedocs.io/en/latest/install/howto-install-AMUSE.html) and install all the dependancies
* Install AMUSE Framework and [BSE](https://amuse.readthedocs.io/en/latest/reference/available-codes.html#bse) with `pip`
    <br> 
    ```
    pip install amuse-framework
    pip install amuse-bse
    ```
* Clone this repository
    ```
    git clone https://github.com/gautam-404/Binary-Evolution.git
    ```
<br>

### Running the code
* Run 
    ```
    cd Binary-Evolution
    python main.py
    ```
    In case the python program throws an MPI exception (like `MPI_ERR_SPAWN: could not spawn processes`), explicitly use Open-MPI instead of MPICH (refer [this doc](https://amuse.readthedocs.io/en/latest/install/howto-install-AMUSE.html)) and run `echo 'export OMPI_MCA_rmaps_base_oversubscribe=yes' >> ~/.zshrc`. For Bash shell, replace `~/.zshrc` with `~/.bashrc`. Restart your terminal.
    
    
The python code will create an output folder with evolution histories of all binary systems at the path "~/OutputFiles". See /src/BPS_eval.py which contains necessary functions that produce the data to see the total gamma-ray luminosity produced by the population as a function of time. 

<br>

[Contact](mailto:gautam.anuj.1195@gmail.com) for more info.
