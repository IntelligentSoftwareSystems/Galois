#!/bin/bash
#
# This script sets up a development environment on Ubuntu 18.04 and is adapted
# from the CI scripts under .github/workflows. Feel free to adjust these
# instructions for your distribution of choice.

# installing g{cc,++}-9
sudo add-apt-repository ppa:ubuntu-toolchain-r/test 
sudo apt update
sudo apt install gcc-9 g++-9

# installing up-to-date cmake https://apt.kitware.com/
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
  | gpg --dearmor - \
  | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
sudo apt-get update
sudo apt upgrade cmake
# alternatively:
#   pip install cmake

# installing arrow and parquet
curl -fL --output /tmp/arrow-keyring.deb \
  https://apache.bintray.com/arrow/ubuntu/apache-arrow-archive-keyring-latest-$(lsb_release --codename --short).deb \
  && sudo apt install -yq /tmp/arrow-keyring.deb \
  && rm /tmp/arrow-keyring.deb

# installing up-to-date llvm
sudo apt-add-repository 'deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-10 main'
curl -fL https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo apt update
apt install -yq clang-10 clang++-10 clang-format-10 clang-tidy-10 llvm-10-dev

# make clang-{tidy,format}-10 the default
sudo update-alternatives --verbose --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-10 90
sudo update-alternatives --verbose --install /usr/bin/clang-format clang-format /usr/bin/clang-format-10 90
