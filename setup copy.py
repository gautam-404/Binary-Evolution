"""Setup.py module for the workflow's worker utilities.
All the workflow related code is gathered in a package that will be built as a
source distribution, staged in the staging area for the workflow being run and
then installed in the workers when they start running.
This behavior is triggered by specifying the --setup_file command line option
when running the workflow for remote execution.
"""

from distutils.command.build import build as _build
import subprocess
import os 
os.system('apt install build-essential gfortran \
  libopenmpi-dev openmpi-bin \
  libgsl-dev cmake libfftw3-3 libfftw3-dev \
  libgmp3-dev libmpfr6 libmpfr-dev \
  libhdf5-serial-dev hdf5-tools \
  libblas-dev liblapack-dev')

import setuptools


# This class handles the pip install mechanism.
class build(_build):  # pylint: disable=invalid-name
  """A build command class that will be invoked during package install.
  The package built using the current setup.py will be staged and later
  installed in the worker using `pip install package'. This class will be
  instantiated during install for this specific scenario and will trigger
  running the custom commands specified.
  """
  sub_commands = _build.sub_commands + [('CustomCommands', None)]


# Some custom command to run during setup. The command is not essential for this
# workflow. It is used here as an example. Each command will spawn a child
# process. Typically, these commands will include steps to install non-Python
# packages. For instance, to install a C++-based library libjpeg62 the following
# two commands will have to be added:
#
#     ['apt-get', 'update'],
#     ['apt-get', '--assume-yes', install', 'libjpeg62'],
#
# First, note that there is no need to use the sudo command because the setup
# script runs with appropriate access.
# Second, if apt-get tool is used then the first command needs to be 'apt-get
# update' so the tool refreshes itself and initializes links to download
# repositories.  Without this initial step the other apt-get install commands
# will fail with package not found errors. Note also --assume-yes option which
# shortcuts the interactive confirmation.
#
# The output of custom commands (including failures) will be logged in the
# worker-startup log.
CUSTOM_COMMANDS = [
    ['apt', 'update'],
    ['apt', '--assume-yes', 'install', 'build-essential'],
    ['apt', '--assume-yes', 'install', 'gfortran'],
    ['apt', '--assume-yes', 'install', 'libopenmpi-dev'],
    ['apt', '--assume-yes', 'install', 'openmpi-bin'],
    ['apt', '--assume-yes', 'install', 'libgsl-dev'],
    ['apt', '--assume-yes', 'install', 'cmake'],
    ['apt', '--assume-yes', 'install', 'libfftw3-3'],
    ['apt', '--assume-yes', 'install', 'libfftw3-dev'],
    ['apt', '--assume-yes', 'install', 'libgmp3-dev'],
    ['apt', '--assume-yes', 'install', 'libmpfr6'],
    ['apt', '--assume-yes', 'install', 'libmpfr-dev'],
    ['apt', '--assume-yes', 'install', 'libhdf5-serial-dev'],
    ['apt', '--assume-yes', 'install', 'hdf5-tools'],
    ['apt', '--assume-yes', 'install', 'libblas-dev'],
    ['apt', '--assume-yes', 'install', 'liblapack-dev'],
    ['pip', 'install', 'amuse-framework'],
    ['pip', 'install', 'amuse-bse']
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
REQUIRED_PACKAGES = ['numpy', 'tqdm', 'amuse-framework', 'amuse-bse']


setuptools.setup(
    name='BPS',
    version='1.0',
    # cmdclass={
    #     # Command class instantiated and run during pip install scenarios.
    #     'build': build,
    #     'CustomCommands': CustomCommands,
    #     },
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PACKAGES
    )




