#!/bin/bash

GALOIS_HOME=/net/ohm/export/cdgc/cxh/GaloisCpp
LONESTARGNN=$GALOIS_HOME/build-gnn-cpu/lonestargnn
GNNS="gcn"
GRAPHS="cora citeseer pubmed flickr reddit"
#GRAPHS="cora"
EPOCHS="200"
NTHREADS="56"
DROPOUT="0.1 0.2 0.3 0.5"
LEARNINGRATES="0.01"
HIDDENDIM="16 32 64 128 256 512"
OUTDIR=/net/ohm/export/cdgc/cxh/outputs/DeepGalois

for GNN in $GNNS; do
	for NT in $NTHREADS; do
		for GR in $GRAPHS; do
			for K in $EPOCHS; do
				for DR in $DROPOUT; do
					for LR in $LEARNINGRATES; do
						for HD in $HIDDENDIM; do
							EXEC_DIR=$LONESTARGNN/$GNN
							echo $EXEC_DIR
							echo "$EXEC_DIR/$GNN $GR -k=$K -t=$NT -d=$DR -lr=$LR -h=$HD &> $OUTDIR/$GNN-$GR-$K-$DR-$LR-$HD-$NT.log"
							$EXEC_DIR/$GNN $GR -k=$K -t=$NT -d=$DR -lr=$LR -h=$HD &> $OUTDIR/$GNN-$GR-$K-$DR-$LR-$HD-$NT.log
							echo "Done. Check out $OUTDIR/$GNN-$GR-$K-$DR-$NT.log"
						done
					done
				done
			done
		done
	done
done
