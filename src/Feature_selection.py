import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

class Feature_Selectors():

    def selectkbest(X_train, X_test, y_train, n=10):
        selector = SelectKBest(score_func=f_classif, k=n)
        X_train_k = selector.fit_transform(X_train, y_train)
        X_test_k = selector.transform(X_test)
        return X_train_k, X_test_k

    def apply_lda(X_train,X_test,y_train,n):
        lda = LDA(n_components = n)
        X_train_lda = lda.fit_transform(X_train, y_train)
        X_test_lda = lda.transform(X_test)
        return X_train_lda, X_test_lda

    def apply_pca(X_train, X_test, n):
        pca = PCA(n_components=n)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        explained_variance = pca.explained_variance_ratio_
        return X_train_pca, X_test_pca, explained_variance

    
    