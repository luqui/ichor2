

while (my $line = <>) {
    if ($line =~ /<<(\w+)/) {
        my $endtoken = $1;
        my @lines;
        my $indent;
        
        my $going = 1;
        while (<>) {
            chomp;
            if (/^(\s*)$endtoken\s*$/) {
                $indent = $1;
                $going = 0;
                last;
            }
            else {
                s/([\\"])/\\$1/g;
                push @lines, "$_\\n\\\n";
            }
        }

        die "Reached end of file when scanning for end of heredoc $endtoken" if $going;

        s/^$indent// for @lines;
        my $str = join '', @lines;
        $line =~ s/<<$endtoken/"$str"/;
    }
    print $line;
}
