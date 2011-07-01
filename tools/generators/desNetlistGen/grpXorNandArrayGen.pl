#!/usr/bin/perl

use strict;
use warnings;

my $N = shift @ARGV;

print "finish=100000\n\n";


foreach my $i ( 0..$N-1 ) {

   print "\ngroup\n";
   print "\ninputs a$i, b$i end\n";
   print "\noutputs o$i end \n";

   print "\ninitlist a$i\n";
   foreach my $j( 0..$N*$N-1 ) {
      print 5*$j, ", ", ($j%2),"\n"; # toggle value after every 5 
   }
   # 0,0
   # 5,1
   # 10,0
   # 15,1
   print "end\n";

   print "\ninitlist b$i\n";
   foreach my $j ( 0..$N*$N-1 ) {
      print 10*$j, ", ", ($j%2),"\n"; # toggle value after every 10 
   }
   print "end\n";

   # the gate netlist
   print <<HERE;

netlist 
nand2(x$i,a$i,b$i)#40
nand2(y$i,a$i,x$i)#40
nand2(z$i,b$i,x$i)#40
nand2(o$i,y$i,z$i)#40
end
HERE

   #end of group i
   print "\nend\n";

}
