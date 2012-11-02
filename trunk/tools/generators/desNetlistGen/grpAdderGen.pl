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

genAdderGrped("adder$numBits.grp", $numBits, 2**$numBits-1, 2**$numBits-1, 1);

sub genAdderGrped {
   my ($fileName,$numBits,$a,$b,$cin)=@_;

   # open(OUTFILE, "> $fileName" ) or die "Could not open file $fileName for writing";
   *OUTFILE = *STDOUT;

   print OUTFILE "\nfinish $FinishTime\n";
  
   my $aInitArray = genInitListNbit($numBits, "a", $a);
   my $bInitArray = genInitListNbit($numBits, "b", $b);
   my $cinInit = genInitList1bit("cin", $cin);

   for my $i ( 0..$numBits-1 ) {
      print OUTFILE "\ngroup\n";

      # input declaration
      print OUTFILE "\ninputs a$i, b$i";
      if( $i == 0 ) {
         print OUTFILE ", cin ";
      }
      print OUTFILE " end\n";

      #input initlist 
      print OUTFILE $aInitArray->[$i];
      print OUTFILE $bInitArray->[$i];

      if( $i == 0 ) {
         print OUTFILE $cinInit;
      }


      print OUTFILE "outputs s$i";
      if( $i == $numBits-1 ) {
         print OUTFILE ", cout ";
      }
      print " end\n";

      # instantiate 1 bit adder here
      my $cinNet = ($i==0) ? "cin": "c".($i-1);
      my $coutNet = ($i==$numBits-1) ? "cout" : "c$i";

      print OUTFILE "\nnetlist\n";

      print OUTFILE (genAdder1bit( "s$i", $coutNet, "a$i", "b$i", $cinNet ));

      print OUTFILE "end\n";


      print OUTFILE "\nend\n";
   }

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
