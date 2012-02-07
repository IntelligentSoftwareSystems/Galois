#!/usr/bin/perl

use strict;

print "*** Collect vtune profiles. Run as: run.py -- runvtune.pl output app args*\n";

exit 1 unless ($#ARGV > 1);

shift @ARGV;

my $threads = shift @ARGV;
my $outfile = shift @ARGV;
my $num_args = $#ARGV + 1;
my $vtune = "/opt/intel/vtune_amplifier_xe_2011/bin64/amplxe-cl";
my $cmdline = join(" ", @ARGV) . " -t $threads";

print "*** Executing: " . $cmdline . "\n";

system("rm -r tmp.vtune.r$threads");
system("mkdir tmp.vtune.r$threads");
system("$vtune -collect nehalem_general-exploration -result-dir=tmp.vtune.r$threads -start-paused -- $cmdline");
system("echo THREADS\t$threads >>$outfile.line.log");
system("$vtune -R hw-events -r tmp.vtune.r$threads -group-by source-line -csv-delimiter tab >> $outfile.line.log");
system("echo THREADS\t$threads >>$outfile.function.log");
system("$vtune -R hw-events -r tmp.vtune.r$threads -group-by function -csv-delimiter tab >> $outfile.function.log");
