
plots/hierarchical/dendogram_all_columns.svg: src/hierarchical/__init__.py input/normalized.csv
	mkdir -p $(dir $@)
	python $< all-columns

hierarchical_plots: plots/hierarchical/dendogram_all_columns.svg

all: hierarchical_plots


.PHONY: all hierarchical_plots
