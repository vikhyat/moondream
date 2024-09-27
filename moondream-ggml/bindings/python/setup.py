#!/usr/bin/env python

import os
import subprocess

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

HERE = os.path.dirname(os.path.realpath(__file__))

class CMakeBuildExt(build_ext):
    def build_extensions(self):
        import platform
        import sys
        import sysconfig
        import pybind11
        
        # Work out the relevant Python paths to pass to CMake.
        ext = self.extensions[0]
        if platform.system() == "Windows":
            cmake_python_library = "{}/libs/python{}.lib".format(
                sysconfig.get_config_var("prefix"),
                sysconfig.get_config_var("VERSION"),
            )
            if not os.path.exists(cmake_python_library):
                cmake_python_library = "{}/libs/python{}.lib".format(
                    sys.base_prefix,
                    sysconfig.get_config_var("VERSION"),
                )
        else:
            cmake_python_library = "{}/{}".format(
                sysconfig.get_config_var("LIBDIR"),
                sysconfig.get_config_var("INSTSONAME"),
            )
        cmake_python_include_dir = sysconfig.get_path("include")
        
        #install_dir = os.path.abspath(
        #    os.path.dirname(self.get_ext_fullpath("dummy"))
        #)
        #os.makedirs(install_dir, exist_ok=True)
        
        ext_dir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name))
        )
        os.makedirs(ext_dir, exist_ok=True)
        cmake_args = [
            "-DCMAKE_INSTALL_PREFIX={}".format(ext_dir),
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={}".format(ext_dir), 
            "-DPython_EXECUTABLE={}".format(sys.executable),
            "-DPython_LIBRARIES={}".format(cmake_python_library),
            "-DPython_INCLUDE_DIRS={}".format(cmake_python_include_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_PREFIX_PATH={}".format(pybind11.get_cmake_dir()),
            "-DDEBUG_BUILD=OFF",
            "-DMOONDREAM_SHARED_LIBS=ON",
            "-DMOONDREAM_EXE=OFF",
            "-G Unix Makefiles",
        ]
        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call(
            ["cmake", HERE] + cmake_args, cwd=self.build_temp
        )

        # Build all the extensions.
        super().build_extensions()

        # Finally run install.
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "install"],
            cwd=self.build_temp,
        )

    def build_extension(self, ext):
        target_name = ext.name.split(".")[-1]
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", target_name],
            cwd=self.build_temp,
        )

extensions = [
    Extension("moondream_ggml.cpp_ffi", ["./ffi.cpp"]),
]

setup(
    name="moondream-ggml",
    author="M87 Labs",
    package_dir={"": "."},
    packages=find_packages("."),
    include_package_data=True,
    ext_modules=extensions,
    cmdclass={"build_ext": CMakeBuildExt},
)
