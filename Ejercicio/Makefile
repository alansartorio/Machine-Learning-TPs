RM=rm

PANDOC=pandoc

PANDOC_OPTIONS=-t revealjs -s --include-in-header=styles.html -V theme=blood

PANDOC_PDF_OPTIONS=

PLOTS=part_2.svg part_3.svg part_4.svg

$(PLOTS): analyzer.py Datos\ Alimenticios.csv
	SHOW_PLOTS=0 python analyzer.py

%.html: %.md $(PLOTS)
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_PDF_OPTIONS) -o $@ $<

watch:
	ls *.md | entr make

clean:
	$(RM) *.pdf $(PLOTS)

.PHONY: docs clean watch
