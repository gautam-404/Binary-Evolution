#!/bin/sh
sudo port install gcc9 openmpi-gcc9 hdf5 gsl cmake gmp mpfr fftw-3 +gcc9 openblas lapack
sudo port install python39
sudo port select --set mpi openmpi-gcc9-fortran
sudo port select --set gcc mp-gcc9
sudo port select --set python3 python39

# shelltype=`echo $0`

python3 -m venv Amuse-env
. Amuse-env/bin/activate
alias amuse-env='. ~/virtualenvironments/Amuse-env/bin/activate'
pip install --upgrade pip
pip install .
