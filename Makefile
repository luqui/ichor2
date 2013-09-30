all: FluidModule.js

%.js: %.heredoc.js jsheredoc.pl
	perl jsheredoc.pl < $< > $@
	node $@

