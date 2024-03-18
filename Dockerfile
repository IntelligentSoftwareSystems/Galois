ARG BUILD_IMAGE=ubuntu:22.04
FROM --platform=linux/amd64 ${BUILD_IMAGE} AS dev

WORKDIR /tmp

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && \
  apt install -y \
  cmake \
  gcc \
  g++ \
  ccache \
  build-essential \
  make \
  libboost-all-dev \
  libfmt-dev \
  libzstd-dev \
  lsb-release \
  wget \
  software-properties-common \
  gnupg \
  gdb \
  vim \
  git \
  python3 \
  python3-pip \
  unzip \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# setup intel repo for intel-basekit
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
  gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | \
  tee /etc/apt/sources.list.d/oneAPI.list
RUN apt update && \
  apt install -y \
  intel-basekit \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

ENV NINJA_BUILD_VERSION=1.11.1
RUN wget https://github.com/ninja-build/ninja/releases/download/v${NINJA_BUILD_VERSION}/ninja-linux.zip -P /tmp && \
  unzip /tmp/ninja-linux.zip -d /usr/bin && \
  rm /tmp/ninja-linux.zip

ARG IS_CI=true

RUN if [ "${IS_CI}" != "true" ] ; then \
  apt update -y \
  &&  apt install -y \
  vim \
  gdb \
  universal-ctags \
  powerline \
  zsh \
  valgrind \
  sudo \
  doxygen \
  texlive-latex-extra \
  texlive-font-utils \
  &&  apt clean; fi

ARG SRC_DIR=/pando-galois
ARG BUILD_DIR=/pando-galois/dockerbuild
ARG UNAME
ARG UID
ARG GID

RUN if [ "${UNAME}" != "root" ] ; then groupadd -g ${GID} ${UNAME} \
  &&  useradd -ms /bin/bash  -u "${UID}" -g "${GID}" ${UNAME} ; fi

RUN mkdir -p /home/${UNAME} \
  && chown ${UNAME}:${UNAME} /home/${UNAME}

USER ${UNAME}
WORKDIR /home/${UNAME}
ENV BUILD_DIR=${BUILD_DIR}

RUN pip3 install compdb pre-commit cpplint "clang-format>=12.0.1,<17.0.0"

RUN echo "PATH=/home/${UNAME}/.local/bin/:\$PATH" >> /home/${UNAME}/.zshenv

RUN echo "export SRC_DIR=${SRC_DIR}" >> /home/${UNAME}/.bashrc
RUN echo "export BUILD_DIR=${BUILD_DIR}" >> /home/${UNAME}/.bashrc
RUN echo "source /opt/intel/oneapi/setvars.sh > /dev/null" >> /home/${UNAME}/.bashrc

WORKDIR ${SRC_DIR}
