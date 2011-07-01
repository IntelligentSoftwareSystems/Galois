#!/usr/bin/perl


use strict;
use warnings;

require "../scripts/netlistlib.pl";
require "../scripts/devicelib.pl";


my $FinishTime = 10000;

my $numBits = shift @ARGV;


genAdderRC( $numBits, 2**$numBits-1, 1, 2**$numBits-1, 2**$numBits-1, 1 );




sub genAdderRC {

   my ($numBits, $sVal, $coutVal, $aVal, $bVal, $cinVal )=@_;

   *OUTFILE=*STDOUT;

   my $inputAList = genPortVector("a",$numBits);
   my $inputBList = genPortVector("b",$numBits);


   print OUTFILE "\nfinish $FinishTime\n";

   print OUTFILE "\ninputs cin, $inputAList, $inputBList\nend\n";


   my $outputList = genPortVector("s", $numBits );
   print OUTFILE "\noutputs cout, $outputList \nend\n";

   my $sOutValues = genOutValuesNbit( $numBits, "s", $sVal );
   my $coutOutValues = genOutValues1bit ( "cout", $coutVal );

   print "\noutvalues\n";
   print "@$sOutValues\n";
   print ", $coutOutValues\n";
   print "end\n";

   my $aInitList = genInitListNbit( $numBits, "a", $aVal );
   my $bInitList = genInitListNbit( $numBits, "b", $bVal );
   my $cinInitlist = genInitList1bit( "cin", $cinVal );

   print "\n@$aInitList\n";
   print "\n@$bInitList\n";
   print "\n$cinInitlist\n";

   if( $numBits == 1 ) { 
      print OUTFILE "\nnetlist\n";
      print OUTFILE genAdder1bit("s", "cout", "a", "b", "cin" ); 
      print OUTFILE "\nend\n";
   }

   else {
      print OUTFILE "\nnetlist\n";
      print OUTFILE ( genAdder1bit("s0", "c0", "a0", "b0", "cin" ) ); 
      print OUTFILE "\nend\n";

      foreach my $i( 1..$numBits-2 ) {
         print OUTFILE "\nnetlist\n";
         print OUTFILE ( genAdder1bit("s$i", "c$i", "a$i", "b$i", "c".($i-1) )   );
         print OUTFILE "\nend\n";
      }

      print OUTFILE "\nnetlist\n";
      print OUTFILE ( genAdder1bit("s".($numBits-1), "cout", "a".($numBits-1), "b".($numBits-1), "c".($numBits-2) ) );
      print OUTFILE "\nend\n";
   }

}


