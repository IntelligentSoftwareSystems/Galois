#!/usr/bin/perl

use strict;
use Getopt::Std;

sub run_prog {
    my $appstr = shift;

    my %options=();
    getopts("hr:t:s:", \%options);
    
    my $threadcount = 24;
    my $threadstart = 1;
    my $numruns = 9;

    if (defined $options{h}) {
	print "-h      :help\n";
	print "-r num  :run num times\n";
	print "-t tmax :end at tmax threads\n";
	print "-s tmin :start at tmin threads\n";
	exit;
    }
    
    if (defined $options{r}) {
	$numruns = $options{r};
    }
    
    if (defined $options{t}) {
	$threadcount = $options{t};
	print "setting threads ending point to $threadcount\n";
    }
    if (defined $options{s}) {
	$threadstart = $options{s};
	print "setting threads starting point to $threadstart\n";
    }
    
    for(my $i = $threadstart; $i <= $threadcount; $i++) {
	print "THREADS: $i\n";
	my %stats;
	for (my $j = 0; $j < $numruns; $j++) {
	    print "*** Executing: " . "$appstr -t $i" . "\n";
	    system("$appstr -t $i");
	    
	    if ($? == -1) {
		print "failed to execute: $!\n";
	    } elsif ($? & 127) {
		printf "child died with signal %d, %s coredump\n",
		($? & 127),  ($? & 128) ? 'with' : 'without';
	    } else {
		printf "child exited with value %d\n", $? >> 8;
	    } 
	}
    }
}

sub vtune_prog {
    my $appstr = shift;

    my %options=();
    getopts("hr:t:s:", \%options);
    
    my $threadcount = 24;
    my $threadstart = 1;

    if (defined $options{h}) {
	print "-h      :help\n";
	print "-t tmax :end at tmax threads\n";
	print "-s tmin :start at tmin threads\n";
	exit;
    }
    
    if (defined $options{t}) {
	$threadcount = $options{t};
	print "setting threads ending point to $threadcount\n";
    }
    if (defined $options{s}) {
	$threadstart = $options{s};
	print "setting threads starting point to $threadstart\n";
    }
    
    for(my $i = $threadstart; $i <= $threadcount; $i++) {
	print "THREADS: $i\n";
	print "*** Executing: " . "$appstr -t $i" . "\n";
	system("rm -r r$i");
	system("mkdir r$i");
	system("/opt/intel/vtune_amplifier_xe_2011/bin64/amplxe-cl -collect nehalem_general-exploration -result-dir=r$i -start-paused -- $appstr -t $i");
	system("/opt/intel/vtune_amplifier_xe_2011/bin64/amplxe-cl -R hw-events -r r$i -group-by source-line -csv-delimiter tab |perl prune_headers_line.pl > results.line.$i.csv");
	system("/opt/intel/vtune_amplifier_xe_2011/bin64/amplxe-cl -R hw-events -r r$i -group-by function -csv-delimiter tab |perl prune_headers_function.pl > results.function.$i.csv");
    }
}

return 1;
exit;
