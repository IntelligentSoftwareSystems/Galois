#!/usr/bin/perl
#

use strict;
use warnings;

require "../scripts/netlistlib.pl";
require "../scripts/devicelib.pl";

my $finishTime = 100000;

my $numBits = shift @ARGV;
my $numEvents = ( @ARGV ) ? shift @ARGV : 200 ;

genCSAtest($numBits, 2**$numBits-1, 2**$numBits-1, 2**$numBits-1, 2**$numBits-1, 2**$numBits-1 );

sub genCSAtest {

   my ($numBits,$s, $t, $a,$b,$c)=@_;


   print "\nfinish $finishTime\n";



   # generate the port declarations
   my $aInput = genPortVector("a",$numBits);
   my $bInput = genPortVector("b",$numBits);
   my $cInput = genPortVector("c",$numBits);

   print "\ninputs\n";
   print "$aInput\n";
   print "$bInput\n";
   print "$cInput\n";
   print "end\n";


   # generate $numEvents events
   my $aInitArray = genInitListNbit($numBits, "a", $a, $numEvents);
   my $bInitArray = genInitListNbit($numBits, "b", $b, $numEvents);
   my $cInitArray = genInitListNbit($numBits, "c", $c, $numEvents);


   print "\n @$aInitArray \n"; 
   print "\n @$bInitArray \n"; 
   print "\n @$cInitArray \n"; 
   

   my $sOutput = genPortVector("s", $numBits );
   my $tOutput = genPortVector("t", $numBits );

   print "\noutputs\n";
   print "$sOutput\n";
   print "$tOutput\n";
   print "end\n";

   my $sOutValList = genOutValuesNbit( $numBits, "s", $s );
   my $tOutValList = genOutValuesNbit( $numBits, "t", $t );

   print "\noutvalues\n";
   print "@$sOutValList\n";
   print "@$tOutValList\n";
   print "\nend\n";



   genCSA($numBits, "s", "t", "a", "b", "c" );
}



sub genCSA {

   my ($numBits,$s,$t,$a,$b,$c) = @_;

   foreach my $i ( 0..$numBits-1 ) {
      print "\nnetlist\n";

      print  genAdder1bit( $s.$i, $t.$i, $a.$i, $b.$i, $c.$i );

      print "\nend\n";
   }

}

