#!/bin/sh
echo "Choose your operating system: MacOS/Linux  (m/l):"
read os 
if [ "$os" = "l" ]; then
    sudo apt-get install build-essential gfortran python3-dev \
    libopenmpi-dev openmpi-bin \
    libgsl-dev cmake libfftw3-3 libfftw3-dev \
    libgmp3-dev libmpfr6 libmpfr-dev \
    libhdf5-serial-dev hdf5-tools \
    libblas-dev liblapack-dev \
    python3-venv python3-pip git
else
    sudo port install gcc9 openmpi-gcc9 hdf5 gsl cmake gmp mpfr fftw-3 +gcc9 openblas lapack
    sudo port install python39
    sudo port select --set mpi openmpi-gcc9-fortran
    sudo port select --set gcc mp-gcc9
    sudo port select --set python3 python39
fi

python3 setup.py
