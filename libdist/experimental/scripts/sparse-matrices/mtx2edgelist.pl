#!/usr/bin/perl
#
# mtx2edgelist.pl - Convert a Matrix Market matrix to a 0-based edgelist
#
use warnings;
use strict;

while (<>) {
    last unless m/^%/;
}
while (<>) {
    next if m/^%/;
    my ($i,$j,$x) = split /[ \t]/;
    print $i-1, ' ', $j-1, ' ', $x;
}
