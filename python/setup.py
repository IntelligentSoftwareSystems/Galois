import subprocess
from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize

includes = subprocess.check_output(
    ["pkg-config", "boost", "--variable=includedir"], universal_newlines=True
).splitlines()

extensions = [
    Extension(
        "galois.galois",
        ["galois/galois.pyx"],
        language="c++",
        extra_compile_args=["-std=c++14",],
        libraries=["galois_shmem", "numa"],
        library_dirs=["lib"],
        include_dirs=["include", *includes],
    ),
Extension(
        "galois.bfs",
        ["galois/bfs.pyx"],
        language="c++",
        extra_compile_args=["-std=c++14",],
        libraries=["galois_shmem", "numa"],
        library_dirs=["lib"],
        include_dirs=["include", *includes],
    ),
Extension(
        "galois.pagerank",
        ["galois/pagerank.pyx"],
        language="c++",
        extra_compile_args=["-std=c++14",],
        libraries=["galois_shmem", "numa"],
        library_dirs=["lib"],
        include_dirs=["include", *includes],
    ),
Extension(
    "galois.sssp",
    ["galois/sssp.pyx"],
    language="c++",
    extra_compile_args=["-std=c++14",],
    libraries=["galois_shmem", "numa"],
    library_dirs=["lib"],
    include_dirs=["include", *includes],
    ),
]


setup(
    name="galois",
    packages=["galois", "galois.bfs", "galois.pagerank", "galois.sssp"],
    package_data={
        "galois": [
            "*.pxd",
            "cpp/*.pxd",
            "cpp/libgalois/*.pxd",
            "cpp/libgalois/Graph/*.pxd",
            "cpp/libstd/*.pxd",
        ]
    },
    ext_modules=cythonize(extensions),
)
