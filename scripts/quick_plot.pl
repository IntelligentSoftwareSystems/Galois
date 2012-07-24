while (<STDIN>) {
    if (/STAT.*/) {
	my @values = split ',';
	if ($values[2] eq $ARGV[0]) {
	    $v{$values[3]} += $values[4];
	    $n{$values[3]} += 1;
	}
    }
}

#foreach $key (sort {$a <=> $b} keys %v) {
#    print "$key $v{$key} $n{$key}\n";
#}

open GP, "|gnuplot -persist" or die "Can't execute gnuplot";

if (exists $n{1}) {
    $doscale = 1;
} else {
    $doscale = 0;
}

if (scalar @ARGV > 1) {
    print "outputfile (eps) is $ARGV[1]\n";
    open GP, "|gnuplot" or die "Can't execute gnuplot";
    print GP "set terminal postscript enhanced color\n";
    print GP "set output '| ps2pdf - $ARGV[1]'\n";
} else {
    open GP, "|gnuplot -persist" or die "Can't execute gnuplot";
}

print GP "set xlabel \"threads\"\n";
print GP "set ylabel \"$ARGV[0]\"\n";
print GP "set y2label \"Scaling\"\n" if $doscale;
print GP "set y2tics nomirror\n" if $doscale;
print GP "set ytics nomirror\n";
print GP "plot '-' title \"$ARGV[0]\" with lines axis x1y1";
print GP ", '-' title \"scaling\" with lines axis x1y2" if $doscale;
print GP "\n";

foreach $key (sort {$a <=> $b} keys %v) {
    print GP $key . " " . ($v{$key} / $n{$key}) . "\n";
}
print GP "e\n";

if ($doscale) {
    foreach $key (sort {$a <=> $b} keys %v) {
	print GP $key . " " . ($v{1} / $n{1}) / ($v{$key} / $n{$key})  . "\n";
    }
    print GP "e\n";
}
