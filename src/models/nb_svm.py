from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from scipy import sparse
import numpy as np
import config
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1,base='svm'):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs
        if base=='logistic':
            self.classifier=LogisticRegression(C=self.C, n_jobs=self.n_jobs)
        else:
            self.classifier=LinearSVC(C=C,dual=dual)

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)

        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
        x_nb = x.multiply(self._r)
        self._clf = self.classifier.fit(x_nb, y)
        return self

if __name__=='__main__':
    model=NbSvmClassifier(dual=True,C=4.5)
    if not os.path.exists(config.PROCESSED_FILEPATH):
        raise ValueError("processed file doesn't exisit")
    fold=1
    df=pd.read_csv(config.PROCESSED_FILEPATH)
    train_df=df[df[config.KFOLD_COL]!=fold]
    valid_df=df[df[config.KFOLD_COL]==fold]
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    
    X_train = vectorizer.fit_transform(train_df[config.TEXT_COL])
    X_test = vectorizer.transform(valid_df[config.TEXT_COL])
    model.fit(X_train,train_df.toxic.astype(int))
    y_pred_train=model.predict(X_train)
    y_pred_test=model.predict(X_test)
    
    from sklearn.metrics import roc_auc_score,classification_report
    print(roc_auc_score(y_score=y_pred_train,y_true=train_df[config.LABEL]))
    print(roc_auc_score(y_score=y_pred_test,y_true=valid_df[config.LABEL]))
    
    
    
    
    
