import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import itertools
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

import umap

import warnings


warnings.filterwarnings("ignore")

cancer = load_breast_cancer()
features = cancer.feature_names

data = np.c_[cancer.data, cancer.target]
columns = np.append(cancer.feature_names, ["target"])
data_df = pd.DataFrame(data, columns=columns)

X = data_df[data_df.columns[:-1]]
y = data_df.target

# print(data_df.describe().T)

corr_columns = list(filter(lambda c: c.startswith("mean"), data_df.columns))

def plot_class_count():
    print("Plotting class count...")
    cnt_pro = data_df['target'].value_counts()
    plt.figure(figsize=(8,6))
    barplot = sns.barplot(x=cnt_pro.index, y=cnt_pro.values, palette=['red', 'green'])
    plt.ylabel('Cantidad de diagnósticos', fontsize=12)
    plt.xlabel('target', fontsize=12)

    # Set x-tick labels
    barplot.set_xticklabels([' 0 = maligno ' , ' 1 =benigno'])

    # plt.show()
    plt.savefig('plots/class_count.svg')




# plt.rcParams.update({'font.size': 20})

def plot_corr_matrix():
    plt.figure(figsize=(10, 9))
    h = sns.heatmap(data_df[corr_columns].corr(), annot=True, linewidths=1)
    h.set_xticklabels(h.get_xticklabels(), rotation=45)
    h.set_yticklabels(h.get_yticklabels(), rotation=0)
    # plt.show()
    plt.tight_layout()
    plt.savefig('plots/corr_matrix.svg')

def plot_corr_target():
    fig, ax = plt.subplots(figsize = (6,10))

    # Compute correlations and sort by absolute values
    corr = data_df[corr_columns].corrwith(data_df['target']).sort_values(key=abs, ascending=False).to_frame()
    corr.columns = ['target']

    sns.heatmap(corr, annot=True, linewidths=0.5, linecolor='black')
    plt.title('Matriz de correlación de las variables con la clasificación')
    # plt.show()
    plt.tight_layout()
    plt.savefig('plots/corr_target.svg')

def plot_distributions():
    # Define the colors
    colors = ['red', 'green']

    fig, ax = plt.subplots(nrows=10, ncols=3, figsize=(20, 30))
    plt.subplots_adjust(wspace=0.2, hspace=0.4)

    for i in range(len(corr_columns)):
        plt.subplot(10, 3, i+1)
        ax = sns.violinplot(x='target', y=corr_columns[i], data=data_df, palette=colors)
        title = features[i] + ' vs target'
        plt.title(title, fontsize=10)

    # plt.show()
    plt.savefig('plots/violin_plots.svg')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# Apply StandardScaler to scale your features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
pca = PCA(n_components=2, random_state=42)
X_train_PCA = pca.fit_transform(X_train_scaled)
X_test_PCA = pca.transform(X_test_scaled)

# LLE
lle_model = LocallyLinearEmbedding(n_components=2, random_state=42)
X_train_LLE = lle_model.fit_transform(X_train_scaled)
X_test_LLE = lle_model.transform(X_test_scaled)

# Kernel PCA
kpca_model = KernelPCA(n_components=2, kernel='rbf', random_state=42)
X_train_KPCA = kpca_model.fit_transform(X_train_scaled)
X_test_KPCA = kpca_model.transform(X_test_scaled)

# Assuming X is your high-dimensional data
UMAP_model = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2, random_state=42)
X_train_UMAP = UMAP_model.fit_transform(X_train_scaled)
X_test_UMAP = UMAP_model.transform(X_test_scaled)

# Assuming X is your high-dimensional data and y are your labels
UMAP_model_2 = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2, random_state=42)
X_train_UMAP_supervised = UMAP_model_2.fit_transform(X_train_scaled, y=y_train)
X_test_UMAP_supervised = UMAP_model_2.transform(X_test_scaled)


def plot_data(X_, y_, titles, filename='dimensionality_reduction'):
    # Define the colors
    colors = ['red' if label == 0 else 'green' for label in y_]

    # Create figure
    plt.figure(figsize=(16, 16))

    # Loop over each subplot
    for i in range(5):
        plt.subplot(5, 1, i+1)
        scatter = plt.scatter(X_[i][:, 0], X_[i][:, 1], c=colors, edgecolor='k')
        plt.title(titles[i])
        plt.gca().set_facecolor('lightgray')
        red_patch = mpatches.Patch(color='red', label='malignant')
        green_patch = mpatches.Patch(color='green', label='benign')
        plt.legend(handles=[red_patch, green_patch])

    # plt.show()
    plt.savefig(f"plots/{filename}.svg")

# Call the function
X_train_list = [X_train_PCA, X_train_KPCA, X_train_LLE, X_train_UMAP, X_train_UMAP_supervised]
titles = ['PCA', 'KPCA', 'LLE', 'UMAP', 'Supervised UMAP']
plot_data(X_train_list, y_train, titles, 'dimensionality_reduction_train')

# Call the function
X_test_list = [X_test_PCA, X_test_KPCA, X_test_LLE, X_test_UMAP, X_test_UMAP_supervised]
titles = ['PCA', 'KPCA', 'LLE', 'UMAP', 'Supervised UMAP']
plot_data(X_test_list, y_test, titles, 'dimensionality_reduction_test')

def evaluate_model(X_train, y_train, X_test, y_test):
    # Fit the classifier
    clf1 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=12, random_state=43)
    clf1.fit(X_train, y_train)

    # Print the accuracy
    print('Accuracy of Decision Tree classifier on original training set: {:.2f}'.format(clf1.score(X_train, y_train)))
    print('Accuracy of Decision Tree classifier on original test set: {:.2f}'.format(clf1.score(X_test, y_test)))

    # Predict the values
    y_pred = clf1.predict(X_test)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix with seaborn
    plt.figure(figsize=(7,5))
    sns.heatmap(cnf_matrix, annot=True, fmt='d')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    # plt.show()
    plt.savefig('plots/confusion_matrix.svg')

    # Print classification report
    print(classification_report(y_test, y_pred))


def evaluate_and_plot_model(X_train, y_train, X_test, y_test):
    # Fit the classifier
    clf1 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=12, random_state=43)
    clf1.fit(X_train, y_train)

    # Print the accuracy
    print('Accuracy of Decision Tree classifier on original training set: {:.2f}'.format(clf1.score(X_train, y_train)))
    print('Accuracy of Decision Tree classifier on original test set: {:.2f}'.format(clf1.score(X_test, y_test)))

    # Predict the values
    y_pred = clf1.predict(X_test)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix with seaborn
    plt.figure(figsize=(7,5))
    sns.heatmap(cnf_matrix, annot=True, fmt='d')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.show()

    # Print classification report
    print(classification_report(y_test, y_pred))


    x_min, x_max = X_train[:, 0].min() , X_train[:, 0].max()
    y_min, y_max = X_train[:, 1].min() , X_train[:, 1].max()

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Predict on the mesh grid
    Z = clf1.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.figure(figsize=(12,10))
    plt.contourf(xx, yy, Z, alpha=0.8)


    colors = ['red' if label == 0 else 'green' for label in y_train]

    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=colors, edgecolor='k')
    plt.gca().set_facecolor('lightgray')
    red_patch = mpatches.Patch(color='red', label='malignant')
    green_patch = mpatches.Patch(color='green', label='benign')
    plt.legend(handles=[red_patch, green_patch])
    plt.show()

plot_class_count()
plot_corr_matrix()
plot_corr_target()
plot_distributions()

# evaluate_model(X_train, y_train, X_test, y_test)

# evaluate_and_plot_model(X_train_PCA, y_train, X_test_PCA, y_test)
# evaluate_and_plot_model(X_train_KPCA, y_train, X_test_KPCA, y_test)
# evaluate_and_plot_model(X_train_LLE, y_train, X_test_LLE, y_test)
# evaluate_and_plot_model(X_train_UMAP, y_train, X_test_UMAP, y_test)
# evaluate_and_plot_model(X_train_UMAP_supervised, y_train, X_test_UMAP_supervised, y_test)
