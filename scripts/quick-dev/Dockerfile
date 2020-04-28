FROM ubuntu:20.04

RUN apt-get update \
      && apt-get install -qy \
      apt-transport-https \
      ca-certificates \
      curl \
      gnupg \
      software-properties-common \
      && curl -fL https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
      && apt-add-repository -y 'deb http://apt.llvm.org/focal/ llvm-toolchain-focal main' \
      && apt-get update
RUN apt-get install -qy \
      ccache \
      clang++-10 \
      clang-10 \
      clang-format-10 \
      clang-tidy-10 \
      cmake \
      g++-9 \
      gcc-9 \
      git \
      gosu \
      libopenmpi-dev \
      llvm-10-dev \
      python3-pip \
      python-is-python3 \
      && update-alternatives --verbose --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 \
      && update-alternatives --verbose --install /usr/bin/g++ g++ /usr/bin/g++-9 90 \
      && update-alternatives --verbose --install /usr/bin/clang clang /usr/bin/clang-10 90 \
      && update-alternatives --verbose --install /usr/bin/clang++ clang++ /usr/bin/clang++-10 90 \
      && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade --no-cache-dir pip setuptools \
      && pip3 install --no-cache-dir conan==1.24
RUN conan profile new --detect --force default \
    && conan profile update settings.compiler.libcxx=libstdc++11 default \
    && conan remote add kmaragon https://api.bintray.com/conan/kmaragon/conan

# Make it a little more convenient when USER is not root
RUN find /root -type f -print0 | xargs -0 chmod 666 \
      && find /root -type d -print0 | xargs -0 chmod 777
RUN echo "export PS1='\\W$ '" >> /root/.bashrc
ENV HOME=/root
# Yes, allow anyone to run as root with gosu
RUN chmod gu+s `which gosu`

VOLUME /root/.conan/data
VOLUME /root/.ccache
VOLUME /source
WORKDIR /source
