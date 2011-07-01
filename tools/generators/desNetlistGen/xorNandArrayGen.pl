#!/usr/bin/perl

use strict;
use warnings;

my $N = shift @ARGV;

print "inputs ";
foreach my $i( 0..$N-1 ) {
   print "a$i, b$i, ";
}
print "\nend\n\n";


print "outputs ";
foreach my $i( 0..$N-1 ) {
   print "o$i, ";
}
print "\nend\n\n";


print "finish=10000\n\n";


foreach my $i( 0..$N-1 ) {
   print "initlist a$i\n";
   foreach my $j ( 0..$N*$N-1 ) {
      print 5*$j, ", ", ($j%2),"\n"; # toggle value after every 5 
   }
   # 0,0
   # 5,1
   # 10,0
   # 15,1
   print "end\n";
}

foreach my $i( 0..$N-1 ) {
   print "initlist b$i\n";
   foreach my $j ( 0..$N*$N-1 ) {
      print 10*$j, ", ", ($j%2),"\n"; # toggle value after every 10 
   }
   print "end\n";
}


# generate the outvalues
print "\noutvalues\n";
foreach my $i (0..$N-1) {
   print "o$i 0,\n";
}
print "end\n";

foreach my $i( 0..$N-1 ) {
   print <<HERE;
netlist 
nand2(x$i,a$i,b$i)#40
nand2(y$i,a$i,x$i)#40
nand2(z$i,b$i,x$i)#40
nand2(o$i,y$i,z$i)#40
end
HERE

}
