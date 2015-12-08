#!/usr/bin/perl
#
# reorder.pl - Reorder a matrix given an ordering.
#              Supports both edgelist and Matrix Market input files.
#

use warnings;
use strict;

die "Usage: $0 [-1] <matrix> <ordering>\n" unless @ARGV == 2;

my $onebased = 0;
if ( $ARGV[0] eq '-1' ) {
    $onebased = 1;
    shift @ARGV;
    warn "Using one-based input"
}

# Load ordering
open F, '<', $ARGV[1] or die;
# number of nonzeros in tim davis code
my @ordering = ();
my $i = 0;
while (<F>) {
    my $j = int($_);
    die if $j < 0;
    $ordering[$j] = $i;
    $i++;
}
close F;

open F, '<', $ARGV[0] or die;
my @rows = ();
my $matrixmarket = 0;
while ( <F> ) {
    if ( m/^[%#]/ ) {
        if ( m/^%%MatrixMarket/ ) {
            warn "Unknown MatrixMarket format" unless m/coordinate real/;
            warn "Assuming one-based" if !$onebased;
            $onebased = 1;
            $matrixmarket = 1;
        }
        print;
        next;
    }
    elsif ( $matrixmarket == 1 ) {
        warn "Skipping MatrixMarket size info\n";
        $matrixmarket++;
        print;
        next;
    }
    s/[\r\n]//g;
    my @row = split(/\s/, $_);
    $row[0] = $onebased ? $ordering[$row[0]-1] : $ordering[$row[0]];
    $row[1] = $onebased ? $ordering[$row[1]-1] : $ordering[$row[1]];
    (defined($row[0]) && defined($row[1])) or die;
    if ( $onebased ) { $row[0]++; $row[1]++ }
    print join(' ', @row), "\n";
}
close F;
