#!/usr/bin/perl
#

use strict;
use warnings;

require "../scripts/netlistlib.pl";
require "../scripts/devicelib.pl";

my $numBits = shift @ARGV;

my $GaloisBin = '~/projects/galois/trunk/bin';
my $testFile = "/tmp/mult$numBits.net";
my $cmd = "cd $GaloisBin && java eventdrivensimulation.SerialEventdrivensimulation $testFile";





foreach my $aVal ( 0..2**$numBits-1) {
   foreach my $bVal( 0..2**$numBits-1) {

      my $mVal    = ($aVal*$bVal) ;


      print "============= aVal = $aVal, bVal = $bVal,  | mVal = $mVal =============\n" if ( 1 );

      open ( FH, "> $testFile") or die "could not open file $testFile"; 

      # generate the test into the file
      genMultTest( *FH, $numBits, $mVal, $aVal, $bVal);

      close ( FH );

      # exit status of the wait call
      my $exitStatus = system( $cmd );
      # exit status of the cmd
      $exitStatus >>= 8;

      if( $exitStatus != 0 ) {
         system( "cp $testFile $testFile.bak");
         print STDERR "Error in simulation, file backed up as $testFile.bak";
         exit(1);
      }
   }
}

