while (<>) {
    @line = split ',';
    chomp @line;
    $function = shift @line;
    $module = shift @line;
    $proc = shift @line;
    $pid = shift @line;
    print "$module:$function," . join(',', @line) . "\n";
}



