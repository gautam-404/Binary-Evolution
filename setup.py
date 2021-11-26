from distutils.command.build import build as _build
import subprocess

import setuptools


# This class handles the pip install mechanism.
class build(_build):  # pylint: disable=invalid-name
  """A build command class that will be invoked during package install.
  """
  sub_commands = _build.sub_commands + [('CustomCommands', None)]


CUSTOM_COMMANDS = [
    ['sudo', 'apt-get', 'update'],
    ['sudo', 'apt-get', '--fix-broken', 'install'],
    ['sudo', 'apt-get', 'upgrade', '-y'],
    ['sudo', 'apt-get', '-yq', 'install', 'build-essential'],
    ['sudo', 'apt-get', '-yq', 'install', 'gfortran'],
    ['bash', 'pkginstaller.sh', 'openmpi-bin'],
    ['sudo', 'rm', '/var/lib/dpkg/info/openmpi-bin.postinst', '-f'],
    ['sudo', 'apt-get', '--fix-broken', 'install'],
    ['sudo', 'apt-get', '-yq', 'install', 'libopenmpi-dev'],
    ['sudo', 'dpkg', '--configure', 'openmpi-bin'],
    ['sudo', 'apt-get' 'install' '-yf'],
    ['sudo', 'apt-get', '-yq', 'install', 'libgsl-dev'],
    ['sudo', 'apt-get', '-yq', 'install', 'cmake'],
    ['sudo', 'apt-get', '-yq', 'install', 'libfftw3-3'],
    ['sudo', 'apt-get', '-yq', 'install', 'libfftw3-dev'],
    ['sudo', 'apt-get', '-yq', 'install', 'libgmp3-dev'],
    ['sudo', 'apt-get', '-yq', 'install', 'libmpfr6'],
    ['sudo', 'apt-get', '-yq', 'install', 'libmpfr-dev'],
    ['sudo', 'apt-get', '-yq', 'install', 'libhdf5-serial-dev'],
    ['sudo', 'apt-get', '-yq', 'install', 'hdf5-tools'],
    ['sudo', 'apt-get', '-yq', 'install', 'libblas-dev'],
    ['sudo', 'apt-get', '-yq', 'install', 'liblapack-dev'],
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
REQUIRED_PACKAGES = ['numpy', 'amuse-framework', 'amuse-bse', 'matplotlib']


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
