while (<>) {
    @line = split '\t';
    chomp @line;
    $file = shift @line;
    $path = shift @line;
    $line = shift @line;
    $module = shift @line;
    $proc = shift @line;
    $pid = shift @line;
    print "\"$file:$line\"," . join(',', @line) . "\n";
}



