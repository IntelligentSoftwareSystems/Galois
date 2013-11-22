#!/usr/bin/perl

use strict;
use warnings;

sub find_vtune () {
  my $vtune = `which amplxe-cl 2> /dev/null`;
  chomp($vtune);
  if (not -e $vtune) {
    my @vtune_vers = ("2013", "2011");
    foreach my $ver (@vtune_vers) {
      my $base = "/opt/intel/vtune_amplifier_xe_" . $ver;
      if (-e $base and not $vtune) {
        $vtune = $base . "/bin64/amplxe-cl";
        last;
      }
    }
  }
  return $vtune;
}

sub find_kernel_sym () {
  # TODO: fix this path when kernel and library debug symbols get installed
  my $symbol = "/usr/lib/debug/boot/" . `uname -r`;
  chomp($symbol);
  return $symbol;
}

sub arch_is_knc ()  {
  if (exists $ENV{'GALOIS_ARCH_MIC'} and $ENV{'GALOIS_ARCH_MIC'} ne '') {
    return 1;
  }
  return 0;
}

sub find_analysis_type () {
  my $type = "nehalem-memory-access";
  # my $type = "snb-general-exploration";
  # my $type = "nehalem-general-exploration";

  if (arch_is_knc ()) {
    $type = 'knc-general-exploration';
    # $type = 'knc-bandwidth';
  }

  my $sys = `hostname`;
  chomp($sys);
  if ($sys eq "volta") {
    $type = "nehalem_general-exploration";
  }
  return $type;
}

sub report_line ($$$$$$) {
  my ($vtune, $report, $rdir, $threads, $outfile, $maxsec) = @_;

  system("echo \"THREADS\t$threads\" >>$outfile.line.log");

  open(my $syspipe, "ulimit -t $maxsec ; $vtune $report $rdir -group-by source-line |") or die($!);
  open(my $output, ">> $outfile.line.log") or die($!);
  my @header = ();
  my @sums = ();
  my $first_data_column = 0;
  while (<$syspipe>) {
    print $output $_;
    chomp;
    my @tokens = split /\t/;
    if (not @header) {
      @header = @tokens;
      for my $tok (@tokens) {
        if ($tok =~ /Hardware Event Count$/) { last; }
        $first_data_column++;
      }
    } else {
      for my $idx ($first_data_column .. @tokens - 1) {
        $sums[$idx] += $tokens[$idx];
      }
    }
  }
  for my $idx ($first_data_column .. @header - 1) {
    my $label = (split /:/, $header[$idx])[0];
    print "RUN: Variable $label = $sums[$idx]\n";
  }
  close $syspipe;
  close $output;
}

sub report_function ($$$$$$) {
  my ($vtune, $report, $rdir, $threads, $outfile, $maxsec) = @_;

  system("echo \"THREADS\t$threads\" >>$outfile.function.log");
  system("ulimit -t $maxsec ; $vtune $report $rdir -group-by function >> $outfile.function.log ");
}

my $vtune = find_vtune;
my $symbol = find_kernel_sym;
my $type = find_analysis_type;
my $user = `whoami`;
chomp($user);

die("Run as: runvtune.pl [-t N] output app args*") unless ($#ARGV >= 1);

my $threads = 1;
my $found_threads = 0;
if ($ARGV[0] eq "-t") {
  shift @ARGV;
  $threads = shift @ARGV;
  $found_threads = 1;
}

my $outfile = shift @ARGV;
my $cmdline = join(" ", @ARGV);

if ($found_threads) {
  $cmdline = $cmdline . " -t $threads";
}

if (arch_is_knc ()) {
  $cmdline = "ssh -t mic0 $cmdline";
}

print "RUN: CommandLine $cmdline\n";

# my $dire = "/tmp/$user.vtune.r$threads";
my $dire = "/workspace/$user/tmp/vtune--r$threads";
if (system ("mkdir -p $dire") != 0) {
  print "failed to use '$dire' for storing vtune data, trying /tmp\n";
  $dire = "/tmp/$user/vtune--r$threads";

  if (system ("mkdir -p $dire") != 0) {
    die "failed to use '$dire' for storing vtune data, quitting\n";
  }
}

my $rdir = "-result-dir=$dire";

my $report = "-R hw-events -format csv -csv-delimiter tab";
# my $type = "hotspots";

my $collect;
if (1) {
  if (arch_is_knc ()) {
    $collect = "-analyze-system -collect $type ";

  } else {
    $collect = "-analyze-system -collect $type -start-paused";
  }
} else {
  # Manual counter configuration
  my @counters = qw(
    LONGEST_LAT_CACHE.MISS
    OFFCORE_RESPONSE_0.ANY_DATA.REMOTE_DRAM
    OFFCORE_RESPONSE_0.ANY_DATA.REMOTE_CACHE
    OFFCORE_RESPONSE_0.ANY_DATA.LOCAL_CACHE
    OFFCORE_RESPONSE_0.ANY_DATA.LOCAL_DRAM
    );
  $collect = "-collect-with runsa -start-paused -knob event-config=" . join(',', @counters);
}

my $sdir = "-search-dir all=$symbol";
my $maxsec = 100000;

system("rm -rf $dire");
system("mkdir -p $dire");

my $vtune_run_cmd = "$vtune $collect $rdir -- $cmdline";
print "Running: '$vtune_run_cmd'\n";
# system("set -x ; $vtune $collect $rdir $sdir -- $cmdline"); 
system("$vtune_run_cmd");
report_function $vtune, $report, $rdir, $threads, $outfile, $maxsec;
report_line $vtune, $report, $rdir, $threads, $outfile, $maxsec;
