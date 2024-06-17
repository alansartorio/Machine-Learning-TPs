

input/normalized.csv input/numerical.csv: input/api_filled.csv
#input/api_filled.csv: input/raw.csv

input/normalized.csv input/numerical.csv input/api_filled.csv:
	HIDE_PLOTS=1 python src/main.py


ITERATIONS_ALL_COLUMNS = out/k_means/iterations_all_columns.csv
CENTROIDS_ALL_COLUMNS = out/k_means/centroids_all_columns.csv
$(ITERATIONS_ALL_COLUMNS) $(CENTROIDS_ALL_COLUMNS): src/k_means/k_means.py input/normalized.csv
	HIDE_PLOTS=1 python src/k_means/k_means.py all-columns


ITERATIONS_NUMERIC_COLUMNS = out/k_means/iterations_numeric_columns.csv
CENTROIDS_NUMERIC_COLUMNS = out/k_means/centroids_numeric_columns.csv
$(ITERATIONS_NUMERIC_COLUMNS) $(CENTROIDS_NUMERIC_COLUMNS): src/k_means/k_means.py input/normalized.csv
	mkdir -p $(dir $@)
	HIDE_PLOTS=1 python $< numeric-columns

out/k_means/silhouette.csv: src/k_means/silhouette.py input/normalized.csv out/k_means/centroids_all_columns.csv
plots/k_means/silhouette.svg: src/k_means/silhouette_plot.py out/k_means/silhouette.csv
plots/k_means/error_by_k.svg: src/k_means/plots.py out/k_means/iterations_all_columns.csv
plots/k_means/confusion.svg plots/k_means/classification.svg: src/k_means/classify.py out/k_means/centroids_all_columns.csv input/normalized.csv input/numerical.csv

out/k_means/silhouette.csv plots/k_means/silhouette.svg plots/k_means/error_by_k.svg plots/k_means/classification.svg: %:
	mkdir -p $(dir $@)
	HIDE_PLOTS=1 python $<


#$(ITERATIONS_ALL_COLUMNS) $(CENTROIDS_ALL_COLUMNS) $(ITERATIONS_NUMERIC_COLUMNS) $(CENTROIDS_NUMERIC_COLUMNS)



k_means_plots: plots/k_means/error_by_k.svg plots/k_means/silhouette.svg plots/k_means/classification.svg

all: k_means_plots


.PHONY: all k_means_plots
