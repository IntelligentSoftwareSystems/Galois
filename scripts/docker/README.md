# Reproducible development environments

For long term development it is better to set up a development environment on
your host development machine, but if you'd like to get started quickly, this
directory contains a Docker configuration called quick-dev to simplify the
configuration of a new development environment.

This directory also contains a configuration, msan, that provides instrumented
libraries for use with `-fsanitize=memory` (GALOIS_USE_SANITIZER=Memory).

# Building

```bash
docker build -t quick-dev .
```

```bash
docker build -t quick-dev .
docker build -t msan -f Dockerfile.msan .
```

# Using

```bash
run-image.sh
# Or...
IMAGE=msan run-image.sh
/source/scripts/docker/msan/config-galois.sh
```
