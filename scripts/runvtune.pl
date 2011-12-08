#!/usr/bin/perl

use strict;

print "*** Filter script to colledt vtune profiling.  Run via: run.py -- runvtune.pl outputfile ./app/my/benchmark args\n";

shift @ARGV;
my $threads = shift @ARGV;
my $outfile = shift @ARGV;

my $num_args = $#ARGV + 1;

#for(my $i = 0; $i < $num_args; $i++) {
#    print "$i: $ARGV[$i]\n";
#}

my $cmdline = join(" ", @ARGV) . " -t $threads";
print "*** Executing: " . $cmdline . "\n";
system("rm -r tmp.vtune.r$threads");
system("mkdir tmp.vtune.r$threads");
system("/opt/intel/vtune_amplifier_xe_2011/bin64/amplxe-cl -collect nehalem_general-exploration -result-dir=tmp.vtune.r$threads -start-paused -- $cmdline");
system("echo THREADS\t$threads >>$outfile.line.log");
system("/opt/intel/vtune_amplifier_xe_2011/bin64/amplxe-cl -R hw-events -r tmp.vtune.r$threads -group-by source-line -csv-delimiter tab >> $outfile.line.log");
system("echo THREADS\t$threads >>$outfile.function.log");
system("/opt/intel/vtune_amplifier_xe_2011/bin64/amplxe-cl -R hw-events -r tmp.vtune.r$threads -group-by function -csv-delimiter tab >> $outfile.function.log");
