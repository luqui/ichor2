

while (my $line = <>) {
    if ($line =~ /<<(\w+)/) {
        my $endtoken = $1;
        my @lines;
        my $indent;
        
        while (<>) {
            chomp;
            if (/^(\s*)$endtoken\s*$/) {
                $indent = $1;
                break;
            }
            else {
                s/(\\")/\\$1/;
                push @lines, "$_\\\n";
            }
        }

        s/^$indent// for @lines;
        my $str = join '', @lines;
        $line =~ s/<<$endtoken/"$str"/;
    }
    print $line;
}
