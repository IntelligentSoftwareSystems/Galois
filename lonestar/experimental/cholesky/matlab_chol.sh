#!/bin/bash
#
# matlab_chol.sh - Do Cholesky factorization in MATLAB (matlab_chol.m)
#

usage() {
    echo "Usage: $0 [options] <input> [output]"
    echo "Options:"
    echo "  -m[path]      Use MATLAB, optionally at given path"
    echo "  -o[path]      Use Octave, optionally at given path"
    echo "  <input>       Input file in 0-based triplet format"
    echo "  [output]      Output file, defaulting to matlabedges.txt"
    exit 1
}

# Find the default interpreter
INTERPRETER=
INTERPRETER_TYPE=
if type -t matlab > /dev/null; then
    INTERPRETER=matlab
    INTERPRETER_TYPE=m
elif type -t octave > /dev/null; then
    INTERPRETER=octave
    INTERPRETER_TYPE=o
fi

# Run getopt and update the positional parameters
getopt -T > /dev/null
if [ "$?" != 4 ]; then
    echo "This script requires an enhanced version of getopt." >&2
    exit 1
fi
OPTTMP=$(getopt -n matlab_chol.sh 'm::o::h' "$@")
[ $? = 0 ] || usage
eval set -- "$OPTTMP"

# Parse the command-line arguments
while true; do
    case "$1" in
        -m|-o)
            INTERPRETER_TYPE="${1#-}"
            INTERPRETER="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            usage
            ;;
    esac
done
[ "$#" = 1 ] || [ "$#" = 2 ] || usage

# If an interpreter type was specified without a name for the interpreter
if [ -z "$INTERPRETER" ]; then
    if [ "$INTERPRETER_TYPE" = m ]; then
        INTERPRETER=matlab
    else
        INTERPRETER=octave
    fi
fi

# CHOL_DIR via http://stackoverflow.com/a/246128/
CHOL_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
CHOL_CMD="matlab_chol;exit"
export CHOL_INPUT="$1"
export CHOL_OUTPUT="$2"

if [ "$INTERPRETER_TYPE" = m ]; then
    echo "Using $INTERPRETER (MATLAB)"
    export MATLABPATH="$CHOL_DIR"
    "$INTERPRETER" -nojvm -nodesktop -nosplash -r "$CHOL_CMD" < /dev/null &&
        exit 0
elif [ "$INTERPRETER_TYPE" = o ]; then
    echo "Using $INTERPRETER (Octave)"
    trap 'exit 10' INT
    "$INTERPRETER" --path "$CHOL_DIR" --eval "$CHOL_CMD" < /dev/null &&
        exit 0
else
    echo "Can't find either MATLAB or Octave in your path."
fi
exit 1
