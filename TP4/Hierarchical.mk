
out/hierarchical/linkage_all_columns.csv: src/hierarchical/__init__.py input/normalized.csv
plots/hierarchical/dendogram_all_columns.svg: src/hierarchical/plot.py out/hierarchical/linkage_all_columns.csv

out/hierarchical/linkage_all_columns.csv plots/hierarchical/dendogram_all_columns.svg:
	mkdir -p $(dir $@)
	python $< all-columns

#plots/hierarchical/dendogram_all_columns.svg out/hierarchical/linkage_all_columns.csv:

hierarchical_plots: plots/hierarchical/dendogram_all_columns.svg

all: hierarchical_plots


.PHONY: all hierarchical_plots
