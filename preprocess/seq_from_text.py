from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
import sklearn.feature_extraction
import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import config
import pandas as pd

class SeqFromTextConversionSK(BaseEstimator,TransformerMixin):
    
    def __init__(self,oov_token=None,max_len=25):
        self.oov_token=oov_token if oov_token else 1
        self.max_len=max_len
        pass
    
    def fit(self,X,tokenizer=None):
        if tokenizer is None:
            self.tokenizer=sklearn.feature_extraction.text.CountVectorizer()
            #tf.keras.preprocessing.text.Tokenizer(oov_token=self.oov_token)
            self.tokenizer.fit(X)
        else:
            self.tokenizer=tokenizer
        return self

    def text_to_seqs_sklearn(self,X):
        vocabulary=self.tokenizer.vocabulary_
        analyzer=self.tokenizer.build_analyzer()
        seqs=[]
        
        for line in X:
            seq=np.zeros((self.max_len,),dtype=np.int32)
            words=analyzer(line)[:self.max_len]
            for i,word in enumerate(words):
                seq[i]=vocabulary.get(word,-1)+2
            seqs.append(seq)
                
        # seqs=[[vocabulary.get(x,-1)+2 for x in analyzer(line) ] for line in X]
        return np.stack(seqs)

    def transform(self,X):
        seqs=self.text_to_seqs_sklearn(X)
        return seqs #elf.tokenizer.texts_to_sequences(X)

if __name__=='__main__':
    df=pd.read_csv(config.PROCESSED_FILEPATH)
    seq_text=SeqFromTextConversionSK()
    X=seq_text.fit_transform(df[config.PROC_TEXT_COL][:5].astype(str))
    print(X)

