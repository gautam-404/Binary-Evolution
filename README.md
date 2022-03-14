# Binary Population Synthesis with BSE

Read the paper for details: [Millisecond Pulsars from Accretion Induced Collapse naturally explain the Galactic Center Gamma-ray Excess](https://arxiv.org/abs/2106.00222)

## Getting started
<!-- * *Linux users can install this repository as a package by executing `pip install git+https://github.com/gautam-404/Binary-Evolution.git`* -->


### Installation from scratch:
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

### Installing with a shell script:
**Linux and MacOS users** can simply clone this repository and then execute `setup-linux.sh` or `setup-macos.sh`, respectively, to install all prerequisites:
```
git clone https://github.com/gautam-404/Binary-Evolution.git
cd Binary-Evolution
```
```
sh setup-linux.sh
python3 -m venv Amuse-env
. Amuse-env/bin/activate
```
Replace `setup-linux.sh` with `setup-macos.sh` if you are on MacOS.

<br>

### Running the code
* Run 
    ```
    cd Binary-Evolution
    python main.py
    ```
    In case the python program throws an MPI exception (like `MPI_ERR_SPAWN: could not spawn processes`), explicitly use Open-MPI instead of MPICH (refer [this doc](https://amuse.readthedocs.io/en/latest/install/howto-install-AMUSE.html)) and run `echo 'export OMPI_MCA_rmaps_base_oversubscribe=yes' >> ~/.zshrc`. For Bash shell, replace `~/.zshrc` with `~/.bashrc`. Restart your terminal.
    
    
The python code will create an output folder with evolution histories of all binary systems at the path "~/OutputFiles". Consecutively, running `python BPS_eval.py` produces the data necessary to see the total gamma-ray luminosity produced by the population as a function of time. 

<br>
Alternatively, a docker image for this project can be pulled using `docker pull algernon11/bps:ubuntu`. 

[Contact](mailto:anujgautam11@gmail.com) for more info.
