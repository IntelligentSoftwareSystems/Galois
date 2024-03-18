SHELL := /bin/bash

UNAME ?= $(shell whoami)
UID ?= $(shell id -u)
GID ?= $(shell id -g)

BASE_IMAGE_NAME ?= pando-galois
IMAGE_NAME ?= ${UNAME}-${BASE_IMAGE_NAME}
SRC_DIR ?= $(shell pwd)
VERSION ?= $(shell git log --pretty="%h" -1 Dockerfile)

CONTAINER_SRC_DIR ?= /pando-galois
CONTAINER_BUILD_DIR ?= /pando-galois/build
CONTAINER_WORKDIR ?= ${CONTAINER_SRC_DIR}
CONTAINER_CONTEXT ?= default
CONTAINER_OPTS ?=
CONTAINER_CMD ?= bash -l
INTERACTIVE ?= i

BUILD_TYPE ?= RelWithDebInfo

# CMake variables
GALOIS_EXTRA_CMAKE_FLAGS ?= ""
GALOIS_EXTRA_CXX_FLAGS ?= ""

# Developer variables that should be set as env vars in startup files like .profile
GALOIS_CONTAINER_MOUNTS ?=
GALOIS_CONTAINER_ENV ?=
GALOIS_CONTAINER_FLAGS ?=
GALOIS_BUILD_TOOL ?= 'Unix Makefiles'
GALOIS_CCACHE_DIR ?= ${SRC_DIR}/.ccache

dependencies: dependencies-asdf

dependencies-asdf:
	@echo "Updating asdf plugins..."
	@asdf plugin update --all >/dev/null 2>&1 || true
	@echo "Adding new asdf plugins..."
	@cut -d" " -f1 ./.tool-versions | xargs -I % asdf plugin-add % >/dev/null 2>&1 || true
	@echo "Installing asdf tools..."
	@cat ./.tool-versions | xargs -I{} bash -c 'asdf install {}'
	@echo "Updating local environment to use proper tool versions..."
	@cat ./.tool-versions | xargs -I{} bash -c 'asdf local {}'
	@asdf reshim
	@echo "Done!"

hooks:
	@pre-commit install --hook-type pre-commit
	@pre-commit install-hooks

pre-commit:
	@pre-commit run -a

ci-image:
	@${MAKE} docker-image-dependencies
	@docker image inspect galois:${VERSION} >/dev/null 2>&1 || \
	docker --context ${CONTAINER_CONTEXT} build \
	--build-arg SRC_DIR=${CONTAINER_SRC_DIR} \
	--build-arg BUILD_DIR=${CONTAINER_BUILD_DIR} \
	--build-arg UNAME=runner \
  --build-arg UID=1078 \
  --build-arg GID=504 \
	-t galois:${VERSION} \
	--file Dockerfile \
	--target dev .

docker-image:
	@${MAKE} docker-image-dependencies
	@docker image inspect ${IMAGE_NAME}:${VERSION} >/dev/null 2>&1 || \
	docker --context ${CONTAINER_CONTEXT} build \
	--build-arg SRC_DIR=${CONTAINER_SRC_DIR} \
	--build-arg BUILD_DIR=${CONTAINER_BUILD_DIR} \
	--build-arg UNAME=${UNAME} \
	--build-arg IS_CI=false \
  --build-arg UID=${UID} \
  --build-arg GID=${GID} \
	-t ${IMAGE_NAME}:${VERSION} \
	--file Dockerfile \
	--target dev .

docker-image-dependencies:
	@mkdir -p build
	@mkdir -p data
	@mkdir -p .ccache

.PHONY: docker
docker:
	@docker --context ${CONTAINER_CONTEXT} run --rm \
	-v ${SRC_DIR}/:${CONTAINER_SRC_DIR} \
	-v ${GALOIS_CCACHE_DIR}/:/home/${UNAME}/.ccache \
	${GALOIS_CONTAINER_MOUNTS} \
	${GALOIS_CONTAINER_ENV} \
	${GALOIS_CONTAINER_FLAGS} \
	--privileged \
	--workdir=${CONTAINER_WORKDIR} \
	${CONTAINER_OPTS} \
	-${INTERACTIVE}t \
	${IMAGE_NAME}:${VERSION} \
	${CONTAINER_CMD}

run-cmake:
	@cmake \
  -S ${SRC_DIR} \
  -B ${BUILD_DIR} \
	-G ${GALOIS_BUILD_TOOL} \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
	-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	-DUSE_MKL_BLAS=ON \
	-DGALOIS_ENABLE_DIST=ON \
	${GALOIS_EXTRA_CMAKE_FLAGS}

setup: run-cmake

setup-ci: run-cmake

run-tests:
	@ctest --test-dir build -R wmd --verbose
	@ctest --test-dir build -R large-vec --verbose
	@ctest --test-dir build -R compile-lscsr --verbose

# this command is slow since hooks are not stored in the container image
# this is mostly for CI use
docker-pre-commit:
	@docker --context ${CONTAINER_CONTEXT} run --rm \
	-v ${SRC_DIR}/:${CONTAINER_SRC_DIR} --privileged \
	--workdir=${CONTAINER_WORKDIR} -t \
	${IMAGE_NAME}:${VERSION} bash -lc "git config --global --add safe.directory /pando-galois && make hooks && make pre-commit"
