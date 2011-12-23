#!/bin/bash
#
# Make distribution tarball

NAME="Galois-2.1.1"

if [[ ! -e COPYRIGHT ]]; then
  echo "Run this from the root source directory" 1>&2
  exit 1
fi

touch "$NAME.tar.gz" # Prevent . from changing during tar
(svn status | grep '^\?' | sed -e 's/^\? *//'; \
  echo "*.swp"; \
  echo "status.html"; \
  echo "apps/bp"; \
  echo "*~"; \
  echo "$NAME.tar.gz") | \
  tar --exclude-from=- --exclude-vcs --transform "s,^\./,$NAME/," -cz -f "$NAME.tar.gz" .
