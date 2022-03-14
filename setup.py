from distutils.command.build import build as _build
import subprocess

import setuptools

# This class handles the pip install mechanism.
class build(_build):  # pylint: disable=invalid-name
  """A build command class that will be invoked during package install.
  """
  sub_commands = _build.sub_commands + [('CustomCommands', None)]


CUSTOM_COMMANDS = [
    ['sudo apt-get update'.split()],
    ['sudo apt-get -y install build-essentials'.split()],
    ['sudo spt-get -y install gfortran'.split()],
    ['sudo apt-get -y install python3-dev'.split()],
    ['sudo apt-get -y install libopenmpi-dev'.split()],
    ['sudo apt-get -y install openmpi-bin'.split()],
    ['sudo apt-get -y install libgsl-dev'.split()],
    ['sudo apt-get -y install cmake'.split()],
    ['sudo apt-get -y install libfftw3-3'.split()],
    ['sudo apt-get -y install libfftw3-dev'.split()],
    ['sudo apt-get -y install libgmp3-dev'.split()],
    ['sudo apt-get -y install libmpfr6'.split()],
    ['sudo apt-get -y install libmpfr-dev'.split()],
    ['sudo apt-get -y install libhdf5-serial-dev'.split()],
    ['sudo apt-get -y install hdf5-tools'.split()],
    ['sudo apt-get -y install libblas-dev'.split()],
    ['sudo apt-get -y install liblapack-dev'.split()],
    ['sudo apt-get -y install python3-venv'.split()],
    ['sudo apt-get -y install python3-pip'.split()],
    ['sudo apt-get -y install git'.split()]
    # ['pip', 'install', 'amuse-framework'],
    # ['pip', 'install', 'amuse-bse']
]

class CustomCommands(setuptools.Command):
  """A setuptools Command class able to run arbitrary commands."""

  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def RunCustomCommand(self, command_list):
    print('Running command: %s' % command_list)
    p = subprocess.Popen(
        command_list,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # Can use communicate(input='y\n'.encode()) if the command run requires
    # some confirmation.
    stdout_data, _ = p.communicate()
    print('Command output: %s' %stdout_data)
    if p.returncode != 0:
      raise RuntimeError(
          'Command %s failed: exit code: %s' % (command_list, p.returncode))

  def run(self):
    for command in CUSTOM_COMMANDS:
        self.RunCustomCommand(command)

# Configure the required packages and scripts to install.
REQUIRED_PACKAGES = ['numpy', 'docutils','mpi4py', 'h5py', 'wheel', 'scipy', 'astropy', 'jupyter', 'pandas', 'seaborn', 'matplotlib', 'amuse-framework', 'amuse-bse', 'matplotlib', 'tqdm']

setuptools.setup(
    name='BPS',
    version='4.2.0',
    cmdclass={
        # Command class instantiated and run during pip install scenarios.
        'build': build,
        'CustomCommands': CustomCommands,
        },
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PACKAGES
    )
