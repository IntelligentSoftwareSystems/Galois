#!/usr/bin/perl

use strict;
use warnings;
use Getopt::Long;

my $debug = 0;
my $num_nodes = 1000;
my $num_levels = 10;
my $avg_deg = 8;
my $max_weight = 10000;

sub myassert {
  if ($debug) {
    my ($result, $msg) = @_;
    if (not $result) {
      my ($pkg, $file, $line) = caller;
      die "Assertion <<$msg>> failed in package: $pkg, file: $file, line:$line\n"
    }
  }
}

my $usage =<<"TILL_HERE";
./$0 options
  -l number of levels (must be > 1)
  -n number of nodes (must be >= number of levels)
  -d average degree of a node
  -w max edge weight
  -X debug mode
  -h this help message
TILL_HERE

sub parse_cmdline {
  my $show_help = 0;
  my $status = GetOptions (
    'l=i' => \$num_levels,
    'n=i' => \$num_nodes,
    'd=i' => \$avg_deg,
    'w=i' => \$max_weight,
    'X' => \$debug,
    'h' => \$show_help,
  );

  if (!$status or ($ARGV) or $show_help) { die $usage; }
}

sub min ($$) {
  return $_[$_[0] > $_[1]];
}

sub per_level ($$) {
  my ($total, $num_levels) = @_;
  return int (($total + $num_levels - 1) / $num_levels);
}

sub pick_random ($$) {
  my ($beg, $end) = @_;
  myassert ($end > $beg, "wrong interval");
  return $beg + int (rand ($end - $beg));
}

sub add_edges ($$$$$) {
  my ($edges_per_level, $curr_start, $curr_end, $next_start, $next_end) = @_;
  # create edges between [curr_start, curr_end) and [next_start, next_end)

  myassert ($curr_end > $curr_start);
  myassert ($next_end > $next_start);

  my %adj = ();

  # first create edges 1 to 1
  my $curr_diff = $curr_end - $curr_start;
  my $next_diff = $next_end - $next_start;
  myassert ($curr_diff >= $next_diff); # assumes rounding up in per_level
  for (my $i = 0; $i < $curr_diff; ++$i) {
    my $src = $i % $curr_diff + $curr_start + 1; # dimacs node ids start from 1
    my $dst = $i % $next_diff + $next_start + 1;

    $adj{$src}{$dst} = 1;

    my $wt = pick_random (0, $max_weight) + 1;
    print "a $src $dst $wt\n";

    --$edges_per_level; # adjust the target
  }

  # now remaining random edges
  for my $i (1..$edges_per_level) {
    my $src = 0;
    my $dst = 0;

    do { 
      $src = pick_random ($curr_start, $curr_end) + 1; # dimacs node ids start from 1
      $dst = pick_random ($next_start, $next_end) + 1;
    } while ($adj{$src}{$dst});

    $adj{$src}{$dst} = 1;

    my $wt = pick_random (0, $max_weight) + 1;
    print "a $src $dst $wt\n";
  }
}

sub generate () {
  myassert ($num_levels > 1);
  myassert ($num_nodes >= $num_levels);
  myassert ($avg_deg > 0);
  myassert ($max_weight > 0);

  my $seed = 0;
  srand ($seed);

  my $num_edges = $avg_deg * $num_nodes;

  # round up
  my $nodes_per_level = per_level ($num_nodes, $num_levels);
  my $edges_per_level = per_level ($num_edges, $num_levels);

  # re-evaluate
  $num_nodes = $num_levels * $nodes_per_level; # for the first level
  $num_edges = ($num_levels - 1) * $edges_per_level + $nodes_per_level;

  print "p sp $num_nodes $num_edges\n";
  print "c num_levels: $num_levels\n";

  # create edges from first node to all nodes in 2nd level
  for my $i (2..$nodes_per_level+1) {
    my $wt = pick_random (0, $max_weight) + 1;
    print "a 1 $i $wt\n";
  }

  for my $i (0..$num_levels-2) {

    my $curr_start = $i * $nodes_per_level + 1; # due to first node

    myassert (($i + 1) < $num_levels);
    my $next_start = ($i + 1) * $nodes_per_level + 1; # due to first node
    my $next_end = min ($num_nodes, $next_start + $nodes_per_level);

    print "c level=$i, creating edges between [$curr_start,$next_start)".
        " and [$next_start,$next_end)\n";


    add_edges ($edges_per_level, $curr_start, $next_start, $next_start, $next_end);
  }
}

sub main () {
  parse_cmdline;
  generate;
}

main;
