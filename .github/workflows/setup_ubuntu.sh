#!/bin/bash

apt install -yq ccache curl gcc-9 g++-9 libopenmpi-dev
update-alternatives --verbose --install /usr/bin/gcc gcc /usr/bin/gcc-9 90
update-alternatives --verbose --install /usr/bin/g++ g++ /usr/bin/g++-9 90

curl -fL --output /tmp/arrow-keyring.deb https://apache.bintray.com/arrow/ubuntu/apache-arrow-archive-keyring-latest-$(lsb_release --codename --short).deb \
  && apt install -yq /tmp/arrow-keyring.deb \
  && rm /tmp/arrow-keyring.deb
apt update
apt install -yq libarrow-dev libparquet-dev 

apt-add-repository 'deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-10 main'
curl -fL https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
apt update
# Install llvm via apt instead of as a conan package because existing
# conan packages do yet enable RTTI, which is required for boost
# serialization.
apt install -yq clang-10 clang++-10 clang-format-10 clang-tidy-10 llvm-10-dev

update-alternatives --verbose --install /usr/bin/clang++ clang++ /usr/bin/clang++-10 90
update-alternatives --verbose --install /usr/bin/clang clang /usr/bin/clang-10 90
update-alternatives --verbose --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-10 90
update-alternatives --verbose --install /usr/bin/clang-format clang-format /usr/bin/clang-format-10 90

pip3 install --upgrade pip setuptools
pip3 install conan==1.24
