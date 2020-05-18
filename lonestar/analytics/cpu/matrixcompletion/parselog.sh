#!/bin/bash
perl -pe 'if(/(\d+) elapsed \(ms\): (\S+).*GFLOP\/s: (\S+) RMSE.*: (\S+)/){ print "STAT SINGLE Elapsed$1 (null) $2\n";}' \
| perl -pe 'if(/(\d+) elapsed \(ms\): (\S+).*GFLOP\/s: (\S+) RMSE.*: (\S+)/){ print "STAT SINGLE GFLOPS$1 (null) $3\n";}' \
| perl -pe 'if(/(\d+) elapsed \(ms\): (\S+).*GFLOP\/s: (\S+) RMSE.*: (\S+)/){ print "STAT SINGLE RMSE$1 (null) $4\n";}' \
| ~/w/GaloisDefault/scripts/report.py
