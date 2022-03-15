from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

from collections import Counter
import pandas as pd
import numpy as np
import cv2

class CorpusSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns):
        
        self.columns = columns
        
    def fit(self, X, y = None):
        
        return self
    
    def transform(self, X):
    
        X_new = X[self.columns[0]]

        if len(self.columns) > 1:
            for column in self.columns[1:]:
                X_new += '. ' + X[column]
                
        return X_new
    
class ColumnsSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, column:str=None):
        
        self.column = column
        
    def fit(self, X, y = None):
        
        return self
    
    def transform(self, X, y = None):
        
        if not self.column is None:
            X_new = X[self.column].values
        else:
            X_new = X.copy()
        
        return X_new
        
class GloveVectorizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, vectors_dict:dict, method:str='mean', vector_length:int=300):
        """
        Ressource to weighted mean: https://openreview.net/forum?id=SyK00v5xx
        """
        assert method in ['mean', 'sum', 'weighted']
        
        self.vectors_dict = vectors_dict
        self.method = method
        self.vector_length = vector_length

    def fit(self, X, y = None):

        return self

    def transform(self, X, y = None):
        
        X_new = X.copy()

        X_new = X_new.apply(lambda x: [(word, self.vectors_dict[word]) if word in self.vectors_dict else (word, list(np.zeros(self.vector_length))) for word in x])

        if self.method == 'mean':
            X_new = X_new.apply(lambda x: pd.Series(np.mean([vector for word, vector in x], axis=0)))
        
        elif self.method == 'sum':
            X_new = X_new.apply(lambda x: pd.Series(np.sum([vector for word, vector in x], axis=0)))
        
        else:
            corpus = X.explode().values
            frequencies = {word: freq / len(corpus) for word, freq in Counter(corpus).items()}
            X_new = X_new.apply(lambda x: pd.Series(np.sum([(10 ** -3 / (10 ** -3 + frequencies[word])) * np.array(vector) for word, vector in x], axis=0) / np.sum([frequencies[word]  for word, vector in x], axis=0)))
        
        return X_new
    
class UniversalSentenceEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, model):
        
        self.model = model

    def fit(self, X, y = None):
        
        return self

    def transform(self, X, y = None):
        
        X_new = X.copy()
        X_new = np.array(self.model(X_new))
                      
        return X_new
    
class CustomPCA(BaseEstimator, TransformerMixin):
    
    def __init__(self, explained_variance=None, ratio:float=0.99, random_state:int=0):

        self.explained_variance = explained_variance
        self.ratio = ratio
        self.random_state = random_state

    def fit(self, X, y = None):
        
        if not self.ratio is None:
            
            if self.explained_variance is None:
                pca = PCA(random_state = self.random_state)
                X_pca = pca.fit_transform(X)
                self.explained_variance = pca.explained_variance_ratio_
            
            index = np.argwhere(np.cumsum(self.explained_variance) < self.ratio)
            index = index if index.size else np.array([0])
            self.n_components_ = np.argmax(index) + 1
            self.model = PCA(self.n_components_, random_state = self.random_state)
            self.model.fit(X)
        
        return self
    
    def transform(self, X):
        
        if self.ratio is None:
            X_new = X.copy()
        else:
            X_new = self.model.transform(X)
        
        return X_new
    
class CustomORB(BaseEstimator, TransformerMixin):
    
    def __init__(self, n_features:int):
        
        self.n_features = n_features
        self.model = cv2.ORB_create(nfeatures=self.n_features)
    
    def fit(self, X, y = None):
        
        return self
    
    def transform(self, X, y = None):
        
        X_new, self.keypoints_ = list(), list()
        
        for img in X:
            keypoints, descriptors = self.model.detectAndCompute(img, None)
        
            if len(keypoints) == 0:
                descriptors = np.zeros((self.n_features, 32))
            elif len(keypoints) < self.n_features:
                descriptors = np.concatenate([descriptors, np.zeros((self.n_features - len(keypoints), 32))])
            
            self.keypoints_.append(keypoints)
            X_new.append(descriptors)
        
        X_new = np.array(X_new, dtype='object').reshape(len(X) * self.n_features, 32)
        
        return X_new

class ImagesFilter(BaseEstimator, TransformerMixin):
    
    def __init__(self, gaussian:bool=True, median:bool=True):
        
        self.gaussian = gaussian
        self.median = median

    def fit(self, X, y = None):
        
        return self

    def transform(self, X, y = None):
        
        X_new = list()
        
        for img in X:
            if self.median:
                img = cv2.medianBlur(img, 3)
            if self.gaussian:
                img = cv2.GaussianBlur(img, (5, 5), 1)
           
            X_new.append(img)
            
        X_new = np.array(X_new, dtype='object')
        
        return X_new

class BagOfVisualWords(BaseEstimator, TransformerMixin):
    
    def __init__(self, n_clusters:int, n_features:int, random_state:int=0):
        
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.random_state = random_state
        
        self.model = MiniBatchKMeans(n_clusters = self.n_clusters, random_state = self.random_state)
    
    def fit(self, X, y = None):
        
        self.model.fit(X)
        
        return self
    
    def transform(self, X, y = None):

        X_clus = self.model.transform(X)
        X_new = list()
        
        for i in np.arange(0, len(X), self.n_features):
            cluster_count = list(np.bincount(self.model.labels_[i:i + self.n_features]))
            cluster_count = np.array(cluster_count + list(np.zeros(self.n_clusters - len(cluster_count))))
            X_new.append(cluster_count)
        
        X_new = np.array(X_new)
        
        return X_new

class CustomClustering(BaseEstimator, ClassifierMixin):

    def __init__(self, model=None):
        
        self.model = model
    
    def fit(self, X, y):

        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
                
        self.X_ = X
        self.y_ = y
        
        encoder = LabelEncoder()
        self.y_enc_ = encoder.fit_transform(self.y_)
        self.classes_dict_ = {classe:i for i, classe in zip(encoder.classes_, range(len(encoder.classes_)))}

        return self

    def predict(self, X):
        
        check_is_fitted(self)
        X = check_array(X)
        
        if not self.model is None:
            self.model.fit(X)
            C = self.model.labels_
        else:
            C = X.flatten()

        cf = confusion_matrix(self.y_enc_, C)
        l = linear_sum_assignment(cf, maximize=True)
        
        self.classes_dict_pred_ = {i: self.classes_dict_[j] for j, i in zip(range(len(l[1])), l[1])}
        C = np.array([self.classes_dict_pred_[i] for i in C])
        
        return C
        

