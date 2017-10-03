#!/usr/bin/perl
#
# Take output of run_vtune.pl and produce tab-deliminated file

use strict; 
use warnings;
use Getopt::Long;
use Pod::Usage;

# Command line options
my $InType = "line";
my %validInTypes = map { $_ => 1 } ("line", "function");
my $ShowType = "raw";
my %validShowTypes = map { $_ => 1 } ("raw", "ratio", "scalebythread");
my $Help = 0;

my %Stats = ();
my %Thread_keys = ();
my $TOTAL = "TOTAL";

my $DEBUG=0;
sub debug {
  print STDERR ">>>DEBUG: @_\n" if $DEBUG;
}

sub show {
  my ($tmap, $lk, $tk) = @_;

  if (!exists $tmap->{$lk}{$tk}) {
    print STDERR "ERROR: missing key: tk=$tk, lk=$lk\n";
  }

  if ($ShowType eq "scalebythread") {
    return $tmap->{$lk}{$tk} / $tk;
  } elsif ($ShowType eq "ratio") {
    return $tmap->{$lk}{$tk} / $tmap->{$TOTAL}{$tk};
  } elsif ($ShowType eq "raw") {
    return $tmap->{$lk}{$tk};
  } else {
    die;
  }
}

GetOptions('show=s'=>\$ShowType, 'in=s'=>\$InType, 'help'=> \$Help) or pod2usage(2);
pod2usage(-exitstatus=>0, -verbose=>2, -noperldoc=>1) if $Help;
die("unknown show type") unless ($validShowTypes{$ShowType});
die("unknown in type") unless ($validInTypes{$InType});

my $newSet = 0;
my $curThread = 0;
my @heads = ();

while (<>) {
  chomp;
  my @line = split /\t/;

  debug "line:@line\n";
  # print "line:$line[0],$line[1]\n";

  if ($line[0] =~ /^THREADS$/) {
    debug "Threads line: @line\n";
    $newSet = 1;
    $curThread = $line[1];
    $Thread_keys{$curThread} = 1;
  } elsif ($newSet) {
    $newSet = 0;
    @heads = @line;
    debug "headers:@heads\n";
  } else {
    my $ind;
    my $offset = 0;

    debug "line=@line, length=$#line\n";

    if ($InType eq "line") {
      # first 2 columns are source file and line
      my $file = shift @line;
      my $ln = shift @line;

      $offset = 2;
      if ($heads[$offset] =~ /file path/i) {
        my $path = shift @line;
        $offset += 1;
      } 

      $ind = "$file:$ln";

    } elsif ($InType eq "function") {
      # first column is function name 
      # last 4 colums are module, function full name, source file, start address
      my $function = shift @line;

      my $address = pop @line;
      my $file = pop @line;
      my $fullname = pop @line;
      my $module = pop @line;

      $offset = 1;
      $ind = "$file:$fullname:$module:$address";
    }

    debug "line=@line, length=$#line\n";

    for (my $i = 0; $i <= $#line; $i++) {
      my $nk = $heads[$i + $offset];
      debug "nk=$nk\n";
      $Stats{$nk}{$curThread}{$ind} += $line[$i];
      $Stats{$nk}{$curThread}{$TOTAL} += $line[$i];
    }
    # print "###\n";
  }
}

# for the combinations of (line_keys, Thread_keys) for a given stat_name that don't
# have corresponding Stats, we put a 0. e.g. a particular function/line shows
# up in the profile at threads=1 but not at threads=16.
foreach my $nk (keys %Stats) {
  my %line_keys = ();
  foreach my $tk (keys %{$Stats{$nk}}) {
    foreach my $lk (keys %{$Stats{$nk}{$tk}}) {
      $line_keys{$lk} = 1;
    }
  }

  foreach my $tk (keys %Thread_keys) {
    foreach my $lk (keys %line_keys) {
      if (!exists $Stats{$nk}{$tk}{$lk}) {
        $Stats{$nk}{$tk}{$lk} = 0;
      }    
    }
  }
}

my $maxThread = (sort { $a <=> $b } keys %Thread_keys)[-1];

foreach my $nk (sort keys %Stats) {
  print "$nk";
  foreach my $tk (sort { $a <=> $b } keys %Thread_keys) {
    print "\t$tk";
  }
  print "\n";

  my %transpose = ();
  foreach my $tk (keys %Thread_keys) {
    foreach my $lk (keys %{$Stats{$nk}{$tk}}) {
      $transpose{$lk}{$tk} = $Stats{$nk}{$tk}{$lk};
    }
  }

  # delete lines with all 0s from transpose
  foreach my $lk (keys %transpose) {
    my $all_zeros = 1;
    foreach my $tk (keys %Thread_keys) {
      if ($transpose{$lk}{$tk} != 0) {
        $all_zeros = 0;
        last;
      }
    }

    if ($all_zeros) {
      delete $transpose{$lk};
    }
  }

  #sort by final thread performance
  foreach my $lk (sort { show(\%transpose, $b, $maxThread) <=> show(\%transpose, $a, $maxThread) }
    keys %transpose) {

    print "$lk";
    foreach my $tk (sort { $a <=> $b } keys %Thread_keys) {
      print "\t" . show(\%transpose, $lk, $tk);
    }
    print "\n";
  }

  print "\n\n\n";
}

__END__

=head1 NAME

report_vtune - Emit tab-separated file from output of run_vtune

=head1 SYNOPSIS

cat output | report_vtune [options] > output.tsv

 Options:
   -help             brief help message
   -in=INTYPE        format of run_vtune output
   -show=SHOWTYPE    output format

=head1 OPTIONS

=over 8

=item B<-help>

Print a brief help message and exits.

=item B<-in>=INTYPE

run_vtune output is by INTYPE instead function: line, function

=item B<-show>=SHOWTYPE

Output SHOWTYPE instead of raw counts: raw, ratio, scalebythread

=back

=head1 DESCRIPTION

Emit tab-separated file from output of run_vtune

=cut
