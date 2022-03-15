from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

def plot_TSNE(X, y, title:str='TSNE', random_state:int=0, legend=True, axis='on', perplexity=30):
    
    tsne = TSNE(random_state=random_state, perplexity = perplexity)
    X = tsne.fit_transform(X)
    chart_data = pd.DataFrame({'x': X[:, 0], 'y': X[:, 1], 'category': y})
    
    sns.scatterplot(data=chart_data, x='x', y='y', hue='category', legend=legend)
    plt.xticks([])
    plt.yticks([])
    plt.axis(axis)        
    plt.title(title, size=16)
    if legend:
        plt.legend(bbox_to_anchor=(1, 1))
    
class ConfusionMatrix():
    
    def __init__(self, y_true:list, y_pred:list, metric):
        
        self.y_true = y_true
        self.y_pred = y_pred
        self.metric = metric
        
    def _compute(self):
        
        self.labels_ = self.y_true.unique()
        self.matrix_ = confusion_matrix(self.y_true, self.y_pred, labels=self.labels_)  
        
    def __str__(self):
                
        self._compute()

        plt.figure(figsize=(8, 8))
        plt.title(f"Score : {round(self.metric(self.y_true, self.y_pred), 2)}", size=16)
        sns.heatmap(self.matrix_ / self.matrix_.sum(axis=1, keepdims=True), fmt='.1%', annot=True, cmap='Blues', linewidths=2, linecolor="white", cbar=False)
        plt.legend({0: 'a', 1: 'b'}, loc='upper right', fancybox=True)
        
        plt.xticks(ticks=np.arange(0.5, len(self.labels_), 1), labels=self.labels_, rotation=90)
        plt.yticks(ticks=np.arange(0.5, len(self.labels_), 1), labels=self.labels_, rotation=0)
        plt.ylabel('Actual', weight='bold', size=16)
        plt.xlabel('Predicted', weight='bold', size=16)
    
        plt.show()
        
        return ""
    
class ClusterDistribution():
    
    def __init__(self, y_true:list, y_pred:list):
        
        self.y_true = y_true
        self.y_pred = y_pred
        
    def _compute(self):
       
        self.labels_ = list(set(self.y_true))
        self.colors_dict_ = {label:sns.color_palette('pastel')[i] for i, label in enumerate(self.labels_)}
        
        label_dist = pd.Series(self.y_true).value_counts()
        label_pred_dist = pd.Series(self.y_pred).value_counts()
            
        return label_dist, label_pred_dist

        
    def __str__(self):
        
        label_dist, label_pred_dist = self._compute()
        
        plt.figure(figsize=(14, 8))
            
        plt.subplot(121)
        plt.pie(
            x = label_dist.values, labels=label_dist.index, autopct='%1.f %%', startangle=150,  
            pctdistance=0.7, colors = [self.colors_dict_[label] for label in label_dist.index], 
            textprops={'fontsize': 14, 'c':'w', 'weight':'bold'})
        plt.title('True labels', fontdict={'fontsize': 24})
        plt.legend(bbox_to_anchor=(2.9, 0.95))  
        
        plt.subplot(122)
        plt.pie(
            x = label_pred_dist.values, autopct='%1.f %%', startangle=15,  
            pctdistance=0.7, colors = [self.colors_dict_[label] for label in label_pred_dist.index], 
            textprops={'fontsize': 14, 'c':'w', 'weight':'bold'})
        plt.title('Predicted labels', fontdict={'fontsize': 24})

        
        plt.show()
        
        return ""