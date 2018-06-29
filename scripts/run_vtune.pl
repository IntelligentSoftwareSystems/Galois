#!/usr/bin/perl
#
# Run vtune and collect report to file

use strict;
use warnings;
use Getopt::Long;
use Pod::Usage;
use Cwd;

## Search for vtune in common places
sub find_vtune() {
  my $vtune = `which amplxe-cl 2> /dev/null`;
  chomp $vtune;
  if (not -e $vtune) {
    my @vtune_vers = ('', '_2018');
    foreach my $ver (@vtune_vers) {
      my $base = "/opt/intel/vtune_amplifier$ver/bin64/amplxe-cl";
      if (-e $base) {
        return $base;
      }
    }
  }
  return $vtune;
}

sub extra_symbols_option() {
  my $uname = `uname -r`;
  chomp $uname;
  my @sdirs = ();
  my @candidates = ("/usr/lib/debug/boot/$uname");
  foreach my $c (@candidates) {
    if (-e $c) {
      push(@sdirs, "-search-dir all=$c");
    }
  }
  return join('', @sdirs);
}

sub report_dir_option($) {
  my ($threads) = @_;
  my $user = `whoami`;
  chomp $user;
  my $cwd = getcwd();
  my @candidates = ("/workspace/$user/tmp/vtune--r$threads", "/tmp/$user/vtune--r$threads", "$cwd/vtune--r$threads");
  foreach my $c (@candidates) {
    if (system("mkdir -p $c") == 0) {
      return ($c, "-result-dir=$c");
    }
  }
  return ('', '');
}

sub counter_option(@) {
  return ['-collect-with runsa -knob event-config=' . join(',', @_), '-report hw-events'];
}

## returns analysis and report type option together
sub analysis_option(@) {
  my ($vtune, $a) = @_;
  my @candidates = (
    ["$a", "hw-events"],
    [qw/memory-access hw-events/],
    [qw/memory-consumption hw-events/],
    [qw/general-exploration hw-events/],
    [qw/hotspots hotspots/]
  );
  foreach my $pair (@candidates) {
    my $try = `$vtune -collect @$pair[0] 2>&1`;
    unless ($try =~ /Cannot find analysis type/) {
      return ("-collect @$pair[0]", "-report @$pair[1]");
    }
  }
  return ('','');
}

sub report_line($$$$) {
  my ($cmd, $threads, $outfile, $maxsec) = @_;

  system("echo \"THREADS\t$threads\" >>$outfile.line.log");

  open(my $syspipe, "ulimit -t $maxsec ; $cmd -format csv -csv-delimiter tab -group-by source-line |") or die($!);
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
        $first_data_column++;
        last if ($tok =~ /Module$/);
      }
    } else {
      for my $idx ($first_data_column .. @tokens - 1) {
        $sums[$idx] += $tokens[$idx];
      }
    }
  }
  for my $idx ($first_data_column .. @header - 1) {
    my $label = (split /:/, $header[$idx])[1];
    print "RUN: Variable $label = $sums[$idx]\n";
  }
  close $syspipe;
  close $output;
}

sub report_function($$$$) {
  my ($cmd, $threads, $outfile, $maxsec) = @_;

  system("echo \"THREADS\t$threads\" >>$outfile.function.log");
  system("ulimit -t $maxsec; $cmd -format csv -csv-delimiter tab -group-by function >> $outfile.function.log ");
}

my @counters = ();
my $analyzeSystem = 1;
my $startPaused = 1;
my $help = 0;
my $threads = 0;
my $analysisType = 'memory-access'; # causes vtune 2016 to hang
# my $analysisType = 'general-exploration';
my $reportType = 'hw-events';
my $reportTimeout = 100000;
GetOptions(
  't|threads=s'=>\$threads,
  'analysisType=s'=>\$analysisType,
  'reportType=s'=>\$reportType,
  'counter=s'=>\@counters,
  'startPaused!'=>\$startPaused,
  'analyzeSystem!'=>\$analyzeSystem,
  'reportTimeout=s'=>\$reportTimeout,
  'help'=>\$help) or pod2usage(2);
pod2usage(-exitstatus=>0, -verbose=>2, -noperldoc=>1) if $help;
my $outfile = shift @ARGV;
my $cmdline = join(" ", @ARGV);

if ($threads) {
  $cmdline = "$cmdline -t $threads";
}

my $vtune = find_vtune;
my $symbol = extra_symbols_option();
my ($rdir, $rdiropt) = report_dir_option($threads);
my ($copt, $ropt);
if (@counters) {
  ($copt, $ropt) = counter_option(@counters);
} else {
  ($copt, $ropt) = analysis_option($vtune, $analysisType);
}

die("cannot find way to run vtune") unless($rdir and $copt and $ropt);
die("no command given") unless($cmdline);

print "RUN: CommandLine $cmdline\n";

my @collect = ();
push @collect, $vtune, $symbol, $rdiropt, $copt;
push(@collect, '-analyze-system') if ($analyzeSystem);
push(@collect, '-start-paused') if ($startPaused);
push @collect, '--', $cmdline;

my @report = ();
push @report, $vtune, $rdiropt, $ropt;

system("rm -rf $rdir") == 0 or die($!);
system("mkdir -p $rdir") == 0 or die($!);

my $vtune_collect_cmd = join(' ', @collect);
print "Running: '$vtune_collect_cmd'\n";
system("$vtune_collect_cmd") == 0 or die("vtune collection failed");
report_function join(' ', @report), $threads, $outfile, $reportTimeout;
report_line join(' ', @report), $threads, $outfile, $reportTimeout;

__END__

=head1 NAME

run_vtune - run vtune and parse results to file

=head1 SYNOPSIS

report_vtune [options] <outputbasename> <commandline>

  Options:
    -help                  brief help message
    -analysisType=T        specify vtune analysis type manually
    -reportType=T          specify vtune report type manually
    -counter=C             specify hardware performance counters manually
    -startPaused           start vtune paused (default)
    -reportTimeout=SEC     timeout for generating report
    -nostartPaused         start vtune running
    -analyzeSystem         analyze entire system (default)
    -noanalzeSystem        analyze just command and child processes

=head1 OPTIONS

=over 8

=item B<-analysisType>=T

Run "amplxe-cl -help collect" to see which collection methods are available.

=item B<-reportType>=T

Run "amplxe-cl -help report" to see which reports are available.

=item B<-counter>=C

Use multiple options for multiple counters. Examples of counter names are:
LONGEST_LAT_CACHE.MISS or OFFCORE_RESPONSE_0.ANY_DATA.LOCAL_DRAM

=back

=head1 DESCRIPTION

Run vtune and parse results to file

=cut

