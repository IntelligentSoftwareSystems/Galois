#!/usr/bin/perl
#

use strict;
use warnings;


require "../scripts/devicelib.pl";
require "../scripts/netlistlib.pl";



my $numBits = shift @ARGV;

my $aVal = 2**$numBits-1;
my $bVal = 2**$numBits-1;
my $mVal = $aVal*$bVal ;

genMultTest(*STDOUT, $numBits, $mVal, $aVal, $bVal );



