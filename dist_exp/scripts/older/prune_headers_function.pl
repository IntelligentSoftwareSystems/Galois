while (<>) {
    @line = split '\t';
    chomp @line;
    $function = shift @line;
    $function =~ s/,/_/g;
    $module = shift @line;
    $proc = shift @line;
    $pid = shift @line;
    print "\"$module:$function\"," . join(',', @line) . "\n";
}



