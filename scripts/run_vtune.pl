#!/usr/bin/perl

use strict;

# TODO: check for other common places
my $vtune = "/opt/intel/vtune_amplifier_xe_2011/bin64/amplxe-cl";
my $symbol = "/usr/lib/debug/boot/" . `uname -r`;
chomp($symbol);

die("Run as: runvtune.pl [-t N] output app args*") unless ($#ARGV > 1);

my $threads = 1;
if (@ARGV[0] == "-t") {
  shift @ARGV;
  $threads = shift @ARGV;
}

my $outfile = shift @ARGV;
my $cmdline = join(" ", @ARGV) . " -t $threads";

print "*** Executing: " . $cmdline . "\n";

my $rdir = "-result-dir=tmp.vtune.r$threads";
my $report = "-R hw-events -csv-delimiter tab";
my $collect = "-analyze-system -collect nehalem_general-exploration -start-paused";
my $sdir = "-search-dir all=$symbol";

system("rm -rf tmp.vtune.r$threads");
system("mkdir tmp.vtune.r$threads");
system("$vtune $collect $rdir $sdir -- $cmdline");
system("echo THREADS\t$threads >>$outfile.line.log");
system("$vtune $report $rdir -group-by source-line >> $outfile.line.log");
system("echo THREADS\t$threads >>$outfile.function.log");
system("$vtune $report $rdir -group-by function >> $outfile.function.log");
