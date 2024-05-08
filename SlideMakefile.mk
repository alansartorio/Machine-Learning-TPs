RM=rm

PANDOC=pandoc

PANDOC_OPTIONS=-F mermaid-filter --mathjax -t revealjs -s --include-in-header=styles.html --slide-level=2 -V revealjs-url=reveal.js -V theme=blood

PANDOC_PDF_OPTIONS=

PLOTS=plots
DIST_DIR=dist
DIST=$(DIST_DIR)/index.html $(DIST_DIR)/reveal.js $(PLOTS:%=$(DIST_DIR)/%)

dist/$(PLOTS): $(PLOTS)
	ln -s ../$(PLOTS) $(DIST_DIR)/$(PLOTS)

dist/%.html: %.md styles.html
	mkdir -p dist
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_PDF_OPTIONS) -o $@ $<

dist/%: %
	mkdir -p $(dir $@)
	cp -r $< $(dir $@)

dist/reveal.js:
	wget https://github.com/hakimel/reveal.js/archive/master.tar.gz
	mkdir -p dist/reveal.js
	tar -xf master.tar.gz -C dist/reveal.js reveal.js-master/dist/ reveal.js-master/plugin/ --strip-components=1

watch-slides:
	ls *.md | entr make

clean-slides:
	$(RM) -rf dist

build-slides: $(DIST)
	

.PHONY: docs clean watch build
