#!/usr/bin/perl

#use strict;

my $newSet = 1;
my $curThread = 0;

while (<>) {
    @line = split '\t';
    chomp @line;
 
    if ($line[0] =~ m/^THREADS\s*(\d+)$/) {
	$newSet = 1;
	$curThread = $1;
	$thread_keys{$curThread} += 1;
    } else {
	$file = shift @line;
	$line = shift @line;
	$module = shift @line;
	$path = shift @line;
#	$proc = shift @line;
#	$pid = shift @line;

#	print "$newSet @H\n";

	if ($newSet) {
	    $newSet = 0;
	    @H = @line;
	    foreach $hh (@H) {
		$name_keys{$hh} = 1;
	    }
	} else {
	    $ind = "$module:$file:$line";
#	    print "$ind\n";
	    $line_keys{$ind} = 1;
	    for($i = 0; $i < @line; $i++) {
		$stats{$H[$i]}{$ind}{$curThread} += $line[$i];
		$stats{$H[$i]}{"TOTAL"}{$curThread} += $line[$i];
		#print "$H[$i] $ind $L[$i]\n";
	    }
	}
    }
}
$line_keys{"TOTAL"} = 1;
#exit(0);

#curThread is now maxThread
foreach $nk (sort keys %name_keys) {
    print "$nk";
    foreach $tk (sort { $a <=> $b } keys %thread_keys) {
	print ",$tk";
    }
    print "\n";
    #sort by final thread performance
    foreach $lk (sort { $stats{$nk}{$b}{$curThread} <=> $stats{$nk}{$a}{$curThread} } keys %line_keys) {
	print "$lk";
	foreach $tk (sort { $a <=> $b } keys %thread_keys) {
	    print "," . ($stats{$nk}{$lk}{$tk} / $thread_keys{$tk});
	}
	print "\n";
    }
    print "\n\n\n";
}
