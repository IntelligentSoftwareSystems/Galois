#!/usr/bin/perl -w
# An ifdef lint (we have waaaaay to many ifdefs; sometimes we mistype them).

use strict;
use warnings;

my %ok = ( __linux__ => 1,
           __cplusplus => 1,
           __APPLE_CC__ => 1,
           __MACH__ => 1,
           __LP_64__ => 1,
           __TERMINAL_COLORS => 1,
           NDEBUG => 1,
           SCISM_DEBUG => 1,
           PTHREAD_BARRIER_SERIAL_THREAD => 1,
           DMP_MOT_GRANULARITY => 1,
           DMP_MOT_BITS => 1,
           DMP_WB_GRANULARITY => 1,
           DMP_WB_HASHSIZE => 1,
           DMP_resetRound => 1,
           DMP_setState => 1,
         );
my ($f,$n,$l);
my $error = 0;

sub isHeader($) {
  my ($d) = @_;
  return $d eq "0" || $d =~ /_H(_)?$/;
}

for $f ("dmp-common-config.h", "dmp-common-resource.h") {
  $n = 0;
  for $l (`cat $f`) {
    $n++;
    if ($l =~ /obsolete/i) { last; }
    if ($l =~ /#define (\w+)/i) {
      my $d = $1;
      if (!isHeader($d) && ($f eq "dmp-common-config.h" || $d =~ /_ENABLE_/)) {
        $ok{$d} = 1;
      }
    }
  }
}

for $f (`ls *.h *.cpp`) {
  chomp $f;
  if ($f eq "dmp-common-config.h") { next; }
  $n = 0;
  for $l (`cat $f`) {
    $n++;
    while ($l =~ s/defined(\w+)//) {
      my $d = $1;
      if (!isHeader($d) && !$ok{$d}) { print STDERR "$f:$n: $d\n"; $error = 1; }
    }
    if ($l =~ /#if(n?)def\s+(\w+)/) {
      my $d = $2;
      if (!isHeader($d) && !$ok{$d}) { print STDERR "$f:$n: $d\n"; $error = 1; }
    }
    elsif ($l =~ /defined\((\w+)\)/) {
      my $d = $1;
      if (!isHeader($d) && !$ok{$d}) { print STDERR "$f:$n: $d\n"; $error = 1; }
    }
    elsif ($l =~ /#if\s+(\w+)/) {
      my $d = $1;
      if (!isHeader($d) && !$ok{$d}) { print STDERR "$f:$n: $d\n"; $error = 1; }
    }
  }
}

print "OK.\n" if (!$error);

