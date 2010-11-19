#
# Report.pm
#
# This very simple module is simply for keeping report generation
# in the same place. The code is basically a glorified collection
# of print statements
# Licensed under LGPL 2.1 as packaged with libhugetlbfs
# (c) Mel Gorman 2003

package TLBC::Report;
require Exporter;
use vars qw (@ISA @EXPORT);
use strict;
my $verbose;

@ISA    = qw(Exporter);
@EXPORT = qw(&setVerbose &printVerbose &reportPrint &reportOpen &reportClose);

##
# setVerbose - Set the verbose flag
sub setVerbose {
  $verbose = 1;
}

##
# printVerbose - Print debugging messages if verbose is set
# @String to print
sub printVerbose {
  $verbose && print @_;
}

##
# reportPrint - Print a string verbatim to the report
# @string:	String to print
sub reportPrint {
  my ($string) = @_;

  print HTML $string;
}

##
#
# reportOpen - Open a new report
# @filename: Filename of report to open
sub reportOpen {
  my ($filename) = @_;

  open (HTML, ">$filename") or die("Could not open $filename");
}

##
#
# reportClose - Close the report
sub reportClose {
  close HTML;
}

1;
