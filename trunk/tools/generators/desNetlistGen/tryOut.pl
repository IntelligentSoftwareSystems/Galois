#!/usr/bin/perl 
#

use strict;
use warnings;


my $v = [ map { "a_".$_; } (0..10) ];

print "@$v[0..3]\n";
