my $num = 268435456; #28
#my $num = 134217728; #27
#my $num =  67108864; #26
#my $num =  33554432; #25
#my $num =  16777216; #24
#my $num =   8388608; #23
#my $num =   4194304; #22
#my $num =   2097152; #21

my $edge = $num * 4;
my $maxweight = 10000;

print "p sp $num $edge\n";
print "c random4-n graph 2 to the 28\n";

for ($count = 1; $count < $num; ++$count) {
    print "a $count " . ($count + 1) . " " . (int(rand($maxweight)) + 1). "\n";
}
print "a $num 1 " . (int(rand($maxweight)) + 1). "\n";

for ($count = $num; $count < $edge; ++$count) {
    print "a " . (int(rand($num)) + 1) . " " . (int(rand($num)) + 1). " " . (int(rand($maxweight)) + 1). "\n";
}
