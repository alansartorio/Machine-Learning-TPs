
out/hierarchical/linkage_all_columns.csv: src/hierarchical/__init__.py input/normalized.csv
plots/hierarchical/dendogram_all_columns.svg: src/hierarchical/plot.py out/hierarchical/linkage_all_columns.csv
plots/hierarchical/classification.svg plots/hierarchical/confusion.svg: src/hierarchical/classify.py out/hierarchical/linkage_all_columns.csv input/numerical.csv

out/hierarchical/linkage_all_columns.csv plots/hierarchical/dendogram_all_columns.svg plots/hierarchical/classification.svg plots/hierarchical/confusion.svg:
	mkdir -p $(dir $@)
	HIDE_PLOTS=1 python $< all-columns

#plots/hierarchical/dendogram_all_columns.svg out/hierarchical/linkage_all_columns.csv:

hierarchical_plots: plots/hierarchical/dendogram_all_columns.svg plots/hierarchical/classification.svg plots/hierarchical/confusion.svg

all: hierarchical_plots


.PHONY: all hierarchical_plots
