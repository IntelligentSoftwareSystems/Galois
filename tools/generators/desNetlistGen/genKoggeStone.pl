#!/usr/bin/perl
#

use strict;
use warnings;

require "netlistlib.pl";
require "devicelib.pl";

my $numBits = 8;
my $FinishTime = 10000;

$numBits = shift @ARGV;


genKoggeStoneTest( *STDOUT, $numBits, 2**$numBits-1, 1, 2**$numBits-1, 2**$numBits-1, 1 );






