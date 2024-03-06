SHELL := /bin/bash

IMAGE_NAME := pando-galois
VERSION := 0.0.1
CONTAINER_SRC_DIR := /pando-galois

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

docker-image:
	@docker --context default build --build-arg VERSION=${VERSION} \
	--build-arg UNAME=$(shell whoami) \
  --build-arg UID=$(shell id -u) \
  --build-arg GID=$(shell id -g) \
	-t ${IMAGE_NAME}:${VERSION} \
	--file Dockerfile \
	--target build .

docker:
	@docker --context default run --rm -v $(shell pwd)/:${CONTAINER_SRC_DIR} --privileged --workdir=${CONTAINER_SRC_DIR} -it ${IMAGE_NAME}:${VERSION} bash -l

run-cmake:
	@cmake -S . -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DUSE_MKL_BLAS=ON -DGALOIS_ENABLE_DIST=ON
