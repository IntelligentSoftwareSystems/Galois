#!/bin/bash
#
# prepare-matrix.sh - Prepare a matrix for use, starting from Matrix Market
#
# Takes a Matrix Market input matrix and prepares it to be used with
# Cholesky: conversion to different input formats, METIS reordering,
# symbolic factorization, ....
#
set -e
set -x

# Note: these variables may need to be changed
GALOIS="$(readlink -f "$(dirname "$BASH_SOURCE")")/../../.."
GALOIS_BUILD="default_$(hostname -s)"
METISDIR="$GALOIS/../metis-4.0.3" # /workspace/metis-4.0.3

GC="$GALOIS/build/$GALOIS_BUILD/tools/graph-convert/graph-convert"
MTX2EL="$GALOIS/exp/scripts/sparse-matrices/mtx2edgelist.pl"
IPERM2ORDER="$GALOIS/exp/scripts/sparse-matrices/iperm2order.pl"
CHOLESKY="$GALOIS/build/$GALOIS_BUILD/exp/apps/cholesky/Cholesky"
REORDER="$GALOIS/exp/scripts/sparse-matrices/reorder.pl"
DAGCHOLESKY="$GALOIS/exp/apps/cholesky/simple/upcholesky.serial"
# Note: CSPARSE_SYMBOLIC is CSparse Cholesky modified to do symbolic
# factorization only. Not in repository. (Faster than my symbolic
# implementation.) http://www.cise.ufl.edu/research/sparse/CSparse/
CSPARSE_SYMBOLIC="$GALOIS/exp/apps/cholesky/nha/csparse-inefficient-symbolic"

ONMETIS="$METISDIR/onmetis"
OEMETIS="$METISDIR/oemetis"

TEMPDIR="$(mktemp -d)"
WORKDIR="$(pwd)"

for mtx in "$@"; do
    cd "$WORKDIR"
    mtx="$(readlink -f "$mtx")"
    cd "$TEMPDIR"
    dir="$(dirname "$mtx")"
    matname="$(basename "$mtx" .mtx)"
    base="$dir/$matname"
    gr="$mtx.pyfree.gr"
    edgelist="$base.edgelist"
    sgr="$base.sgr"
    metisgraph="$base.metis"
    iperm="$metisgraph.iperm"
    # Convert MTX file to Galois graph
    "$GC" -mtx2gr -edgeType=float64 "$mtx" "$gr"
    ## Convert Galois graph to edgelist
    #"$GC" -gr2edgelist -edgeType=float64 "$gr" "$edgelist" # BROKEN BROKEN
    "$MTX2EL" < "$mtx" > "$edgelist"
    # Convert Galois graph to METIS input file
    "$GC" -gr2sgr "$gr" "$sgr"
    "$GC" -gr2metis "$sgr" "$metisgraph"
    # For each METIS (onmetis and oemetis)
    for METIS in "$ONMETIS"; do
        # SKIP OEMETIS SKIP OEMETIS # "$OEMETIS"; do
        metisname="$(basename "$METIS")"
        ipermout="$metisgraph.$metisname.iperm"
        ordering="$metisgraph.$metisname.ordering"
        runfilledges="$dir/$matname-$metisname-reordered.txt"
        runfillgr="$dir/$matname-$metisname-reordered.gr"
        fillededges="$dir/fillededges-$matname-$metisname.txt"
        rfillededges="$dir/fillededges-$matname-$metisname-reordered.txt"
        rfilledgr="$dir/fillededges-$matname-$metisname-reordered.gr"
        rfilledrevedges="$dir/fillededges-$matname-$metisname-reordered.revtxt"
        rfilledrevgr="$dir/fillededges-$matname-$metisname-reordered.revgr"
        rcholeskyedges="$dir/dagcholeskyedges-$matname-$metisname-reordered.txt"

        if [ "$metisname" = "onmetis" ]; then
            for f in "$fillededges" "$rfillededges" "$rfilledrevgr" \
                "$rcholeskyedges"; do
                fmetis="${f/-onmetis/-metis}"
                [ -L "$fmetis" ] || ln -s $(basename "$f") "$fmetis"
            done
        fi

        # Run METIS to get a inverse permutation; convert it to an ordering
        [ -x "$METIS" ]
        "$METIS" "$metisgraph" || true
        mv "$iperm" "$ipermout"
        "$IPERM2ORDER" < "$ipermout" > "$ordering"

        # Unfilled reordered files
        "$REORDER" "$edgelist" "$ordering" > "$runfilledges"
        "$GC" -edgelist2gr -edgeType=float64 "$runfilledges" "$runfillgr"

        # Do symbolic factorization
        #"$CHOLESKY" -ordering=fileorder -skipNumeric \
        #    -orderingFile="$ordering" "$gr"
        #mv fillededges.txt "$fillededges"
        "$CSPARSE_SYMBOLIC" < "$runfilledges"
        mv csparse-filled.txt "$rfillededges"

        # DON'T DO NUMERIC FACTORIZATION (also skipping OEMETIS)
        #continue

        # Reorder the filled edges
        #"$REORDER" "$fillededges" "$ordering" > "$rfillededges"
        # Do Cholesky factorization for known result
        "$DAGCHOLESKY" <(seq 0 "$(sort -nu "$ordering"|tail -1)") < "$rfillededges"
        #"$DAGCHOLESKY" < "$rfillededges"
        mv dagcholeskyedges.txt "$rcholeskyedges"

        # Create gr for MUMPS
        "$GC" -edgelist2gr -edgeType=float64 "$rfillededges" "$rfilledgr"
        # Create revgr for Galoized UpCholesky
        awk '{print $2, $1, $3 }' "$rfillededges" > "$rfilledrevedges"
        "$GC" -edgelist2gr -edgeType=float64 "$rfilledrevedges" "$rfilledrevgr"
    done
done
cd "$WORKDIR"
rm -r "$TEMPDIR"
