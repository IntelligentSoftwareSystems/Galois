import sys
import os
import setuptools

from skbuild import setup

# Require pytest-runner only when running tests
pytest_runner = (
    ["pytest-runner>=2.0,<3dev"] if any(arg in sys.argv for arg in ("pytest", "test")) else []
)

setup_requires = pytest_runner


def find_files(root, suffix):
    """
    Find files ending with a given suffix in root and its subdirectories and
    return their names relative to root.
    """
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if not f.endswith(suffix):
                continue
            relpath = os.path.relpath(dirpath, root)
            files.append(os.path.join(relpath, f))
    return files


def package_setup():
    with open("config/version.txt") as f:
        version = f.read().strip()

    pxd_files = find_files("python/galois", ".pxd")

    # "pip wheel --build-option=..." disables use of wheels for dependencies.
    # In order to support passing build arguments directly, accept arguments
    # via the environment.
    cmake_args = os.environ.get("GALOIS_CMAKE_ARGS", "").split()

    # Following PEP-518, use pyproject.toml instead of setup(setup_requires=...) to
    # specify setup dependencies.

    setup(
        version=version,
        name="galois",
        packages=setuptools.find_packages("python"),
        package_data={"galois": pxd_files},
        package_dir={"": "python"},
        tests_require=["pytest"],
        setup_requires=setup_requires,
        cmake_args=cmake_args,
        cmake_source_dir="python",
    )


if __name__ == "__main__":
    package_setup()
