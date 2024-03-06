# Contributing

Contributors must run quality checks on code.  In place of CI we
recommend using `pre-commit` (described below) instead of running
tools like `clang-format` manually.

Code should be clear and documented where needed.

## Setup

Users can run `make docker-image` to setup all dependecies needed for
`pando-galois`.  After creating the image it can be run via `make docker`.
And for first time cmake users can run `make run-cmake`.

# Instrumentation

This section pertains to enabling and instrumenting memory accesses for
performance projections on the theoretical PANDO hardware.

In order for the instrumentation code in `libwmd/include/galois/wmd/instrument.h`,
the following should be added to your top level source directory:

```cmake
set(GALOIS_ENABLE_INSTRUMENT ON)
if (GALOIS_ENABLE_INSTRUMENT)
  add_definitions(-DGALOIS_INSTRUMENT)
endif()
```

Here is a description of the control-flow macros used by the instrumentation
and when they should be used.

```cpp
// Should be called once at the start of the program to initialize the instrumentation
// For example specifying `GRAPH_NAME=example-graph` will result in instrumentation
// files starting with `example-graph`
I_INIT(GRAPH_NAME, HOST, NUM_HOSTS, NUM_EDGES)
// Should be called once at the end of the program to cleanup the instrumentation
I_DEINIT()
// Should be called after the first kernel measured if multiple kernels are being measured
// For example if you specified `GRAPH_NAME=example-graph` above then specifying here that
// `NAME_SUFFIX=-kernel2` will result in instrumentation files starting `example-graph-kernel2`
I_NEW_FILE(NAME_SUFFIX, NUM_EDGES)
// I_ROUND should be called at the end of a communication round to log all memory accesses
// and communication recorded into instrumentation files
// I_CLEAR should be called after I_ROUND
I_ROUND(ROUND_NUM)
I_CLEAR()
// Should be called when sending custom communication to a remote host, recommended practice
// is to just pass in the size of the SendBuffer you are using
I_LC(REMOTE_HOST, BYTES)
```

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
