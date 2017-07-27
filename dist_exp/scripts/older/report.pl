use List::Util qw(sum reduce);

#Gather
my $curthread = 0;

while(<>) {
    if (/RUN: Variable Threads = (\d+)/) {
	$curthread = $1;
    }
    if (/STAT SINGLE (\w+)\s+\(null\)\s+(\d+)/) {
	push(@{$stats{$curthread}{"$1"} }, $2);
	$k{"$1"} = 1;
    }
}

#output
foreach my $th (sort { $a <=> $b } keys %stats) {
    print ",$th";
} 
print "\n";
foreach my $st (sort keys %k) {
    print "$st";
    foreach my $th (sort { $a <=> $b } keys %stats) {
	@values = @{$stats{$th}{$st}};
	if (@values) {
	    my $avg = sum(@values)/@values;
	    print ",$avg";
	} else {
	    print ",0";
	}
    }
    print "\n";
    print "$st Stdev";
    foreach my $th (sort { $a <=> $b } keys %stats) {
	@values = @{$stats{$th}{$st}};
	if (@values) {
	    my $avg = sum(@values)/@values;
	    my $stdev = reduce {$a + ($b - $avg) * ($b - $avg)} 0, @values;
	    $stdev = $stdev / @values;
	    $stdev = sqrt($stdev);
	    print ",$stdev";
	} else {
	    print ",0";
	}
    }
    print "\n";
}

