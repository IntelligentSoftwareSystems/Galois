#!/usr/bin/perl


use strict;
use warnings;

require "../scripts/netlistLib.pl";
require "../scripts/devicelib.pl";


srand(time);

my $MAX_TIME=100;
my $FinishTime = 10000;
my $NUM_THREADS = 4;

my $DEBUG=0;


my $numBits = shift @ARGV;

$NUM_THREADS = (@ARGV) ? shift @ARGV : $NUM_THREADS;


testAdderRC($numBits);


sub testAdderRC {

   my ($numBits) = @_;

   foreach my $a ( 0..(2**$numBits-1) ) {
      foreach my $b ( 0..(2**$numBits-1)  ) {
         foreach my $cin( 0,1 ) {

            my $initAList = genInitList( $a, $numBits, "a" );
            my $initBList = genInitList( $b, $numBits, "b" );
            my $initCinList = genInitList( $cin, 1 , "cin" );


            my $fileName = "../testInputs/adder$numBits.net";

            my @initListArray = ($initAList, $initBList, $initCinList);
            genAdderRC( $fileName , $numBits, \@initListArray );

            my $output=`cd ../bin/ && java -ea main/Main  $fileName $NUM_THREADS`;

            my @lines = split /\n/, $output;

            @lines = grep /^OUTPUT/, @lines;

            if($DEBUG) { print "no. of lines = $#lines\n" }
            if($DEBUG) { print "lines = @lines\n"; }
            

            my $result = scanOutput( \@lines , $numBits );


            print "TEST: a = $a , b = $b, cin = $cin, result = $result,  perl calculated = " , ($a+$b+$cin) , "\n";

            if( ($a + $b + $cin) != $result ) {
               print STDERR  "Test failed with a = $a , b = $b , cin = $cin , result = $result \n";
               print STDERR "output = \n";
               print STDERR $output;

               system("cp $fileName $fileName.faulty");
               die "TEST FAILED";
            }

         } 
      }
   }
}




sub genAdderRC {

   my ($fileName,$numBits, $initListArray )=@_;

   open(OUTFILE, "> $fileName" ) or die "Could not open file $fileName for writing";

   print OUTFILE "\nfinish $FinishTime\n";

   my $inputAList = genPortVector("a",$numBits);
   my $inputBList = genPortVector("b",$numBits);
   print OUTFILE "\ninputs cin, $inputAList, $inputBList\nend\n";


   my $outputList = genPortVector("s", $numBits );
   print OUTFILE "\noutputs cout, $outputList \nend\n";


   foreach my $i ( @$initListArray ) {
      print OUTFILE "\n@$i\n";
   }

   if( $numBits == 1 ) { 
      print OUTFILE "\nnetlist\n";
      print OUTFILE genAdder1bit("s", "cout", "a", "b", "cin" ); 
      print OUTFILE "\nend\n";
   }

   else {
      print OUTFILE "\nnetlist\n";
      print OUTFILE genAdder1bit("s0", "c0", "a0", "b0", "cin" ); 
      print OUTFILE "\nend\n";

      foreach my $i( 1..$numBits-2 ) {
         print OUTFILE "\nnetlist\n";
         print OUTFILE genAdder1bit("s$i", "c$i", "a$i", "b$i", "c".($i-1) );
         print OUTFILE "\nend\n";
      }

      print OUTFILE "\nnetlist\n";
      print OUTFILE genAdder1bit("s".($numBits-1), "cout", "a".($numBits-1), "b".($numBits-1), "c".($numBits-2) ); 
      print OUTFILE "\nend\n";
   }

}


sub genPortVector {
   my ($prefix,$numBits) = @_;

   my $str="";
   my $sep = "";

   if( $numBits == 1 ) {
      $str = $prefix;
   }
   else {
      foreach my $i ( 0..$numBits-1 ) {
         $str =   $str.$sep."${prefix}$i";
         $sep=", ";
      }
   }

   return $str;
}


sub genInitList {

   my ($val, $numBits,$prefix) = @_;


   my $binStr = sprintf("%0${numBits}b",$val);

   if( $DEBUG ) { print "$binStr\n"; }

   my @initList;

   if( $numBits == 1 ) {
      my $rt = int( rand()*$MAX_TIME + 1 ) ; # preventing time to be zero
      push @initList , "initlist $prefix 0,0 $rt, $val end \n";
   }
   else {
      foreach my $i( 0..$numBits-1 ) {
         my $bit = substr( $binStr , length($binStr)-$i-1 , 1 );

         my $rt = int( rand()*$MAX_TIME + 1 ) ; # preventing time to be zero

         push @initList , "initlist ${prefix}$i 0,0 $rt, $bit end \n";
      }
   }

   return \@initList;
}




sub scanOutput {
   my ($lines , $numBits ) = @_;

   print "lines = $lines , numBits = $numBits\n" if $DEBUG;


   my $outNameIndex = 6;
   my $outValIndex = 8;

   my @sumArray = (1..$numBits);
   my $coutVal = 0;
   print "cout = $coutVal , sum = ", reverse( @sumArray ) ,"\n#\n" if $DEBUG;

   foreach my $l ( @$lines ) {

      my @a = split ' ', $l;


      my $outName = $a[$outNameIndex];
      my $outVal = $a[$outValIndex];

      if( $outName =~ /cout/ ) {
         $coutVal = $outVal;
      }
      elsif ( $numBits == 1 ) {
         $sumArray[0] = $outVal;
      }
      else {
         $outName =~ s/\D+//g; # extracting the index
         $sumArray[$outName] = $outVal;
      }
   }

   # DEBUG
   print "cout = $coutVal , sum = ", reverse (@sumArray) ,"\n" if $DEBUG;

   my $result = $coutVal;
   foreach my $i ( reverse @sumArray ) { # msb to lsb
      $result = $result*2;
      $result = $result + $i;
   }

   return $result;
}
