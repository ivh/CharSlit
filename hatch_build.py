import os
import sys
import sysconfig
import subprocess
from pathlib import Path
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        """Build the nanobind extension without CMake."""
        import nanobind

        # Get compilation settings
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
        nanobind_dir = Path(nanobind.__file__).parent
        src_dir = Path("charslit")
        include_dirs = [
            sysconfig.get_path("include"),
            nanobind.include_dir(),
            str(nanobind_dir / "ext" / "robin_map" / "include"),  # For tsl/robin_map.h
            str(src_dir),
        ]

        # Compile C file first as object file
        c_compiler = os.environ.get("CC", "cc")
        c_obj = str(src_dir / "extract.o")

        c_compile_cmd = [
            c_compiler,
            "-c",
            "-fPIC",
            "-O3",
        ]

        for inc in include_dirs:
            c_compile_cmd.extend(["-I", inc])

        c_compile_cmd.extend([str(src_dir / "extract.c"), "-o", c_obj])

        print(f"Compiling C: {' '.join(c_compile_cmd)}")
        result = subprocess.run(c_compile_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"Failed to compile C file: {result.stderr}")

        # Now compile and link the C++ wrapper with the C object
        cxx_compiler = os.environ.get("CXX", "c++")
        output = f"charslit/_charslit{ext_suffix}"

        cxx_compile_cmd = [
            cxx_compiler,
            "-shared",
            "-fPIC",
            "-O3",
            "-std=c++17",
            "-fvisibility=hidden",
        ]

        for inc in include_dirs:
            cxx_compile_cmd.extend(["-I", inc])

        # Add nanobind combined source (includes all other nanobind sources)
        nanobind_src_dir = Path(nanobind.__file__).parent / "src"
        nb_combined = nanobind_src_dir / "nb_combined.cpp"
        cxx_compile_cmd.append(str(nb_combined))

        cxx_compile_cmd.extend([str(src_dir / "extract_wrapper.cpp"), c_obj])
        cxx_compile_cmd.extend(["-o", output])

        # Platform-specific flags
        if sys.platform == "darwin":
            cxx_compile_cmd.extend(["-undefined", "dynamic_lookup"])
        elif sys.platform.startswith("linux"):
            cxx_compile_cmd.append("-Wl,--strip-all")

        print(f"Linking extension: {' '.join(cxx_compile_cmd)}")
        result = subprocess.run(cxx_compile_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"Failed to link extension: {result.stderr}")

        # Force include the extension
        if "force_include" not in build_data:
            build_data["force_include"] = {}

        build_data["force_include"][output] = output

        # __init__.py already exists in charslit/ directory, make sure it's included
        init_file = src_dir / "__init__.py"
        if not init_file.exists():
            init_content = 'from ._charslit import extract\n\n__all__ = ["extract"]\n'
            init_file.write_text(init_content)
        build_data["force_include"][str(init_file)] = str(init_file)
