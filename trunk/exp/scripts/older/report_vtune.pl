while (@ARGV) {
    my $arg = shift @ARGV;
    #work out thread id
    $arg =~ /\w+\.(\d+)\.\w+/;
    $thread = $1;
    $thread_keys{$thread} = 1;
    #print "$arg: $thread\n";
    
    #open file
    open FILE, "<$arg";
    $h = <FILE>;
    @H = split ',', $h;
    foreach $hh (@H) {
	$name_keys{$hh} = 1;
    }
    while ($l = <FILE>) {
	@L = split ',', $l;
	chomp @L;
	$line_keys{$L[0]} = 1;
	for($i = 1; $i < @L; $i++) {
	    $stats{$H[$i]}{$L[0]}{$thread} = $L[$i];
	    #print "$H[$i] $L[0] $L[$i]\n";
	}
    }
}

foreach $nk (sort keys %name_keys) {
    print "$nk";
    foreach $tk (sort { $a <=> $b } keys %thread_keys) {
	print ",$tk";
    }
    print "\n";
    foreach $lk (sort keys %line_keys) {
	print "$lk";
	foreach $tk (sort { $a <=> $b } keys %thread_keys) {
	    print "," . $stats{$nk}{$lk}{$tk};
	}
	print "\n";
    }
    print "\n\n\n";
}
