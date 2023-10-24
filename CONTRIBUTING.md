# Contributing

Contributors must run quality checks on code.  In place of CI we
recommend using `pre-commit` (described below) instead of running
tools like `clang-format` manually.

Code should be clear and documented where needed.

## Setup

Users can run `make docker-image` to setup all dependecies needed for
`pando-galois`.  After creating the image it can be run via `make docker`.
And for first time cmake users can run `make run-cmake`.

## Tools

### [asdf](https://asdf-vm.com)

Provides a declarative set of tools pinned to
specific versions for environmental consistency.

These tools are defined in `.tool-versions`.
Run `make dependencies` to initialize a new environment.

### [pre-commit](https://pre-commit.com)

A left shifting tool to consistently run a set of checks on the code repo.
Our checks enforce syntax validations and formatting.
We encourage contributors to use pre-commit hooks.

```shell
# install all pre-commit hooks
make hooks

# run pre-commit on repo once
make pre-commit
```
