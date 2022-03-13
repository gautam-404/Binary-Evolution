from distutils.command.build import build as _build
import subprocess

import setuptools
from setuptools import setup, find_packages
from support.setup_codes import setup_commands
from support.misc import find_data_files

name = 'amuse-devel'
author = 'The AMUSE team'
author_email = 'info@amusecode.org'
license_ = "Apache License 2.0"
url = 'http://www.amusecode.org/'
install_requires = [
    'wheel>=0.32',
    'docutils>=0.6',
    'numpy>=1.2.2',
    'pytest>=4.0',
    'mpi4py>=1.1.0',
    'h5py>=1.1.0',
]
description = 'The Astrophysical Multipurpose Software Environment'
with open("README.md", "r") as fh:
    long_description = fh.read()
long_description_content_type = "text/markdown"
classifiers = [
    'Development Status :: 4 - Beta',
    'Environment :: Console',
    'Intended Audience :: End Users/Desktop',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: MacOS :: MacOS X',
    'Operating System :: POSIX',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: C',
    'Programming Language :: C++',
    'Programming Language :: Fortran',
    'Topic :: Scientific/Engineering :: Astronomy',
]

extensions = []

all_data_files = find_data_files(
    'data', 'share/amuse/data', '*', recursive=True)
all_data_files.append(('share/amuse', ['./config.mk', './build.py']))

packages = find_packages('src')
packages.extend(['amuse.examples.' + x for x in find_packages('examples')])

package_data = {
    'amuse.rfi.tools': ['*.template'],
    'amuse.test.suite.core_tests': [
        '*.txt', '*.dyn', '*.ini',
        '*.nemo',
        '*.dat', 'gadget_snapshot'
    ],
    'amuse.test.suite.codes_tests': [
        '*.txt', 'test_sphray_data*'
    ],
    'amuse.test.suite.ticket_tests': [
        '*.out'
    ],
    'amuse': [
        '*rc'
    ]
}

mapping_from_command_name_to_command_class = setup_commands()

setup(
    name=name,
    use_scm_version={
        "write_to": "src/amuse/version.py",
    },
    setup_requires=['setuptools_scm'],
    classifiers=classifiers,
    url=url,
    author_email=author_email,
    author=author,
    license=license_,
    description=description,
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    install_requires=install_requires,
    cmdclass=mapping_from_command_name_to_command_class,
    ext_modules=extensions,
    package_dir={'': 'src', 'amuse.examples': 'examples'},
    packages=packages,
    package_data=package_data,
    data_files=all_data_files,
    scripts=["bin/amusifier", "bin/amuse-tutorial", ],
    python_requires=">=3.5"
)


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
