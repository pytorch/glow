from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from distutils.spawn import find_executable
from distutils import sysconfig, log
import setuptools
import setuptools.command.build_py
import setuptools.command.develop
import setuptools.command.build_ext

from collections import namedtuple
from contextlib import contextmanager
import glob
import os
import shlex
import subprocess
import sys
import argparse
from textwrap import dedent
import multiprocessing

try:
    import torch
except ImportError as e:
    print('Unable to import torch. Error:')
    print('\t', e)
    print('You need to install pytorch first.')
    sys.exit(1)


FILE_DIR = os.path.realpath(os.path.dirname(__file__))
TOP_DIR = os.path.realpath(os.path.dirname(FILE_DIR))
CMAKE_BUILD_DIR = os.path.join(TOP_DIR, 'build')

CMAKE = find_executable('cmake') or find_executable('cmake3')
if not CMAKE:
    print('Could not find "cmake". Make sure it is in your PATH.')
    sys.exit(1)

install_requires = []
setup_requires = []
tests_require = []
extras_require = {}

# ################################################################################
# # Flags
# ################################################################################


parser = argparse.ArgumentParser()
parser.add_argument("--debug", type=bool, default=False, help="Compile with debug on")
parser.add_argument("--cmake_prefix_path", type=str, help="Populates -DCMAKE_PREFIX_PATH")
args = parser.parse_known_args()[0]


# ################################################################################
# # Utilities
# ################################################################################


@contextmanager
def cd(path):
    if not os.path.isabs(path):
        raise RuntimeError('Can only cd to absolute path, got: {}'.format(path))
    orig_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)


# ################################################################################
# # Customized commands
# ################################################################################


class cmake_build(setuptools.Command):
    """
    Compiles everything when `python setup.py develop` is run using cmake.

    Custom args can be passed to cmake by specifying the `CMAKE_ARGS`
    environment variable.
    """
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def _run_cmake(self):
        with cd(CMAKE_BUILD_DIR):
            cmake_args = [
                CMAKE,
                '-DGLOW_BUILD_PYTORCH_INTEGRATION=ON',
                '-DBUILD_SHARED_LIBS=OFF',
                '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                '-DCMAKE_BUILD_TYPE={}'.format('Debug' if args.debug else 'Release'),
                '-DPYTHON_EXECUTABLE={}'.format(sys.executable),
                # PyTorch cmake args
                '-DPYTORCH_DIR={}'.format(
                    os.path.dirname(os.path.realpath(torch.__file__))),
            ]

            if args.cmake_prefix_path:
                cmake_args.append('-DCMAKE_PREFIX_PATH={}'.format(args.cmake_prefix_path))

            if 'CMAKE_ARGS' in os.environ:
                extra_cmake_args = shlex.split(os.environ['CMAKE_ARGS'])
                log.info('Extra cmake args: {}'.format(extra_cmake_args))
                cmake_args.extend(extra_cmake_args)
            cmake_args.append(TOP_DIR)
            subprocess.check_call(cmake_args)

    def _run_build(self):
        with cd(CMAKE_BUILD_DIR):
            build_args = [
                CMAKE,
                '--build', os.curdir,
                '--', '-j', str(multiprocessing.cpu_count()),
            ]
            subprocess.check_call(build_args)

    def run(self):
        is_initial_build = not os.path.exists(CMAKE_BUILD_DIR)
        if is_initial_build:
            os.makedirs(CMAKE_BUILD_DIR)
        self._run_cmake()
        self._run_build()


class develop(setuptools.command.develop.develop):
    def run(self):
        self.run_command('build_ext')
        setuptools.command.develop.develop.run(self)


class build_ext(setuptools.command.build_ext.build_ext):
    def run(self):
        self.run_command('cmake_build')
        setuptools.command.build_ext.build_ext.run(self)

    def build_extensions(self):
        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = os.path.basename(self.get_ext_filename(fullname))

            src = os.path.join(CMAKE_BUILD_DIR, "torch_glow", "src", filename)
            dst = os.path.join(os.path.realpath(self.build_lib), 'torch_glow', filename)
            print("dst", dst)
            if not os.path.exists(os.path.dirname(dst)):
                os.makedirs(os.path.dirname(dst))
            self.copy_file(src, dst)


cmdclass = {
    'cmake_build': cmake_build,
    'develop': develop,
    'build_ext': build_ext,
}

# ################################################################################
# # Extensions
# ################################################################################

ext_modules = [
    setuptools.Extension(
        name=str('torch_glow._torch_glow'),
        sources=[])
]

# ################################################################################
# # Packages
# ################################################################################

# # no need to do fancy stuff so far
packages = setuptools.find_packages()

# ################################################################################
# # Test
# ################################################################################

setup_requires.append('pytest-runner')
tests_require.append('pytest')

# ################################################################################
# # Final
# ################################################################################

setuptools.setup(
    name="torch_glow",
    description="PyTorch + Glow",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=packages,
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    author='jackm321',
    author_email='jackmontgomery@fb.com',
    url='https://github.com/pytorch/glow',
)
