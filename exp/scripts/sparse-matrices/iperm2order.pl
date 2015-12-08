#!/usr/bin/perl
#
# iperm2order.pl - Convert an inverted permutation (as used by METIS)
#                  to an ordering (as used by Cholesky).
#
use warnings;
use strict;

my @order = ();
my $i = 0;
while (<>) {
    s/[\r\n]//g;
    $order[$_] = $i;
    $i++;
}
foreach ( @order ) {
    print "$_\n";
}
