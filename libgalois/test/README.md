# Prerequisites

Some tests use sample graphs as inputs, and these can be downloaded with:
```bash
make input
```

If you want to point the tests to an existing set of sample graphs, you
can use the `cmake -DGRAPH_LOCATION=...`.

# Common ctest commands

```bash
# All ctest commands should be run from your build directory
cd ${CMAKE_BINARY_DIR}

# Run all tests in parallel with 4 jobs
ctest -j 4

# Run all tests matching pattern
ctest -R regex

# Run all tests matching label pattern
ctest -L regex

# Show test output
ctest -V

ctest --rerun-failed

# Run tests with valgrind memcheck
ctest --test-action memcheck

# ctest state (e.g., last failed tests, test output) is stored in
# ${CMAKE_BINARY_DIR}/Testing
find Testing/ -type f | xargs cat
```

Tests are divided into several major labels:
- **quick**: Quick tests have no external dependencies and can be run in parallel
  with other quick tests. Each quick test should run in a second or less. These
  tests are run as part of our continuous integration pipeline.
- **nightly**: Nightly tests are tests that take longer (e.g., scalability tests).
  They are run every night on the current master commit.
