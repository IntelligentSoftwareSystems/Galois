#!/usr/bin/perl
#
# Take the output of individual run_vtune commands and merge them into
# a single file

use strict;
use warnings;
use Getopt::Long;
use Pod::Usage;

my $Help = 0;

GetOptions('help'=>\$Help) or pod2usage(2);
pod2usage(-exitstatus=>0, -verbose=>2, -noperldoc=>1) if $Help;
die("need at least one file") unless (@ARGV >= 2);

while (@ARGV) {
  my $threads = shift @ARGV;
  my $filename = shift @ARGV;
  open(my $fh, '<', $filename) or die($!);
  die("empty file") unless (<$fh>);
  print "THREADS\t$threads\n";
  while (my $line = <$fh>) {
    print $line;
  }
}

__END__

=head1 NAME

merge_vtune - Merge output from multiple run_vtune commands

=head1 SYNOPSIS

merge_vtune (<num threads> <file>)+ > merged

=head1 DESCRIPTION

Merge output from multiple run_vtune commands

=cut

