#!/usr/bin/perl

use strict;

# TODO: check for other common places
my $vtune = "/opt/intel/vtune_amplifier_xe_2011/bin64/amplxe-cl";
my $symbol = "/usr/lib/debug/boot/" . `uname -r`;
chomp($symbol);

die("Run as: runvtune.pl [-t N] output app args*") unless ($#ARGV > 1);

my $threads = 1;
my $found_threads = 0;
if (@ARGV[0] eq "-t") {
  shift @ARGV;
  $threads = shift @ARGV;
  $found_threads = 1;
}

my $outfile = shift @ARGV;
my $cmdline = join(" ", @ARGV);

if ($found_threads) {
  $cmdline = $cmdline . " -t $threads";
}

print "*** Executing: " . $cmdline . "\n";

my $uname = `whoami`;
chomp($uname);
# my $type = "nehalem_general-exploration";
my $type = "nehalem_memory-access";

my $sys = `hostname`;
chomp($sys);
if ($sys eq "faraday") {
    $type = "nehalem-memory-access";
}
if ($sys eq "oersted") {
    $type = "nehalem-memory-access";
}

my $dire = "/tmp/$uname.vtune.r$threads";
my $rdir = "-result-dir=$dire";
my $report = "-R hw-events -format csv -csv-delimiter tab";
my $collect = "-analyze-system -collect $type -start-paused";
my $sdir = "-search-dir all=$symbol";
my $maxsec = 1000;

system("date");
system("rm -rf $dire");
system("mkdir $dire");
system("$vtune $collect $rdir $sdir -- $cmdline");
system("echo THREADS\t$threads >>$outfile.line.log");
system("ulimit -t $maxsec ; $vtune $report $rdir -group-by source-line >> $outfile.line.log");
system("echo THREADS\t$threads >>$outfile.function.log");
system("ulimit -t $maxsec ; $vtune $report $rdir -group-by function >> $outfile.function.log");
