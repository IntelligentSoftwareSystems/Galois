ARG BUILD_IMAGE=ubuntu:22.04
FROM --platform=linux/amd64 ${BUILD_IMAGE} AS build

WORKDIR /tmp

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && \
  apt install -y \
  cmake \
  gcc \
  g++ \
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

RUN pip3 install compdb pre-commit cpplint "clang-format>=12.0.1"

RUN echo "PATH=/home/${UNAME}/.local/bin/:\$PATH" >> /home/${UNAME}/.zshenv

RUN echo "export SRC_DIR=${SRC_DIR}" >> /home/${UNAME}/.bashrc
RUN echo "export BUILD_DIR=${BUILD_DIR}" >> /home/${UNAME}/.bashrc
RUN echo "export OMPI_ALLOW_RUN_AS_ROOT=1" >> /home/${UNAME}/.bashrc
RUN echo "export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1" >> /home/${UNAME}/.bashrc
RUN echo "export MKL_ROOT=/opt/intel/oneapi/mkl/2023.2.0" >> /home/${UNAME}/.bashrc

WORKDIR ${SRC_DIR}
