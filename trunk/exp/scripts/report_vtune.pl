#!/usr/bin/perl

use Getopt::Long;
my $inType = "line";
my %validInTypes = map { $_ => 1 } ("line", "function");
my $showType = "raw";
my %validShowTypes = map { $_ => 1 } ("raw", "ratio", "scalebythread");
GetOptions('show=s'=>\$showType, 'in=s'=>\$inType) or die;
die("unknown show type") unless ($validShowTypes{$showType});
die("unknown in type") unless ($validInTypes{$inType});

my $newSet = 1;
my $curThread = 0;

while (<>) {
    my @line = split '\t';
    chomp @line;
 
    if ($line[0] =~ m/^THREADS\s*(\d+)$/) {
	$newSet = 1;
	$curThread = $1;
	$thread_keys{$curThread} += 1;
    } elsif ($newSet) {
        $newSet = 0;
        @H = @line;
        foreach $hh (@H) {
            $name_keys{$hh} = 1;
        }
    } else {
        if ($inType eq "line") {
            $file = shift @line;
            $line = shift @line;
            $module = shift @line;
            $path = shift @line;
            $ind = "$module:$file:$line";
        } elsif ($inType eq "function") {
            $function = shift @line;
            $function =~ s/,/;/g;
            $module = shift @line;
            $ind = "$module:$function";
        }

        $line_keys{$ind} = 1;
        for ($i = 0; $i < @line; $i++) {
            $stats{$H[$i]}{$ind}{$curThread} += $line[$i];
            $stats{$H[$i]}{"TOTAL"}{$curThread} += $line[$i];
        }
    }
}

$line_keys{"TOTAL"} = 1;

my $maxThread = (sort { $a <=> $b } keys %thread_keys)[-1];

foreach $nk (sort keys %name_keys) {
    print "$nk";
    foreach $tk (sort { $a <=> $b } keys %thread_keys) {
	print ",$tk";
    }
    print "\n";

    sub show {
        my $lk = shift;
        my $tk = shift;
        if ($showType eq "scalebythread") {
            return $stats{$nk}{$lk}{$tk} / $tk;
        } elsif ($showType eq "ratio") {
            return $stats{$nk}{$lk}{$tk} / $stats{$nk}{"TOTAL"}{$tk};
        } elsif ($showType eq "raw") {
            return $stats{$nk}{$lk}{$tk};
        }
    }

    #sort by final thread performance
    foreach my $lk (sort { show($b, $maxThread) <=> show($a, $maxThread) } keys %line_keys) {
	print "$lk";
	foreach my $tk (sort { $a <=> $b } keys %thread_keys) {
            print "," . show($lk, $tk);
	}
	print "\n";
    }
    print "\n\n\n";
}
