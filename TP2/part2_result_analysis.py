import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme()

df = pd.read_csv('out/sorted.csv')

def results_for_k(k):
    res = pd.DataFrame()
    for idx, group in df.groupby('original index'):
        expected = group['expected result'].iloc[0]
        actual = group.iloc[:k]['Star Rating'].mode()[0]
        # print('expected', expected)
        # print('actual', actual)
        res = pd.concat([res, pd.DataFrame({
            'expected': [expected],
            'actual': [actual]
        })])

    return res


def plot_confusion_matrix(
        df, 
        true_col, 
        pred_col, 
        figsize=(10, 7), 
        cmap='viridis', 
        show_plot=False,
        save_path='./plots/confusion_matrix.svg'
        ):
    cm = pd.crosstab(df[true_col], df[pred_col], rownames=['True'], colnames=['Predicted'], dropna=False)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='g', cmap=cmap, cbar=False) 
    plt.title('Matriz de confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    if show_plot:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(save_path)
    plt.clf()

    accuracy = {}
    precision = {}
    recall = {}
    f1_score = {}
    
    for label in cm.columns:
        tp = cm.loc[label, label]  # True Positive
        fp = cm[label].sum() - tp  # False Positive
        fn = cm.loc[label].sum() - tp  # False Negative
        tn = cm.sum().sum() - tp - fp - fn  # True Negative
        
        accuracy[label] = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn) != 0 else 0
        precision[label] = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall[label] = tp / (tp + fn) if (tp + fn) != 0 else 0
        if (precision[label] + recall[label]) != 0:
            f1_score[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label])
        else:
            f1_score[label] = 0

    
    return accuracy, precision, recall, f1_score

def plot_metrics(accuracy, precision, recall, f1_score, show_plot=False,save_path='./plots/metrics.svg'):
    metrics_df = pd.DataFrame([accuracy, precision, recall, f1_score])
    metrics_df.index = ['Accuracy', 'Precisión', 'Recall', 'F1-score']

    metrics_df.T.plot(kind='bar', ax=plt.gca())
    plt.title('Precision, Recall, F1-Score per Class')
    plt.ylabel('Score')
    # plt.xlabel('Clase')
    plt.xticks(rotation=45)
    
    if show_plot:
        plt.show()
    else:
        plt.tight_layout(pad=2.0)
        plt.savefig(save_path)
    plt.clf()

r9 = results_for_k(9)
accuracy, precision, recall, f1_score = plot_confusion_matrix(
    r9, 
    'expected', 
    'actual', 
    cmap='Blues',
    save_path='./plots/part_2/confusion_matrix.svg'
    )
plot_metrics(
    accuracy, 
    precision, 
    recall, 
    f1_score,
    save_path='./plots/part_2/metrics.svg'
    )