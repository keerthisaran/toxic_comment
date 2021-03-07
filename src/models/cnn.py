# from ..preprocess import seq_from_text
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
from tensorflow.keras import layers
import sys
import os
from .. import preprocess# import SeqFromTextConversionSK
import config
import pandas as pd

class TF_to_SK(BaseEstimator,ClassifierMixin):
    
    def __init__(self,model,init_params,compile_params,batch_size=64,epochs=10):
        # self.model=compiled_tf_model
        self.batch_size=batch_size
        self.epochs=epochs
        self.model=self.get_tf_model(model,init_params,compile_params)
    

    
    def compile(self,loss,optimizer,metrics=None):
        self.model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
        
    def fit(self,X,y):
        self.model.fit(X,y,batch_size=self.batch_size,epochs=self.epochs)
        
    
    def predict(self,X):
        return self.model(X)

    def get_tf_model(self,model,init_params,compile_params):
        
        init_model=model(**init_params)
        init_model.compile(**compile_params)

        return init_model


class DCNN(tf.keras.Model):
    
    def __init__(self,
                 vocab_size,
                 emb_dim=128,
                 nb_filters=50,
                 FFN_units=512,
                 nb_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="dcnn"):
        super(DCNN, self).__init__(name=name)
        
        self.embedding = layers.Embedding(vocab_size,
                                          emb_dim)
        self.bigram = layers.Conv1D(filters=nb_filters,
                                    kernel_size=2,
                                    padding="valid",
                                    activation="relu")
        self.trigram = layers.Conv1D(filters=nb_filters,
                                     kernel_size=3,
                                     padding="valid",
                                     activation="relu")
        self.fourgram = layers.Conv1D(filters=nb_filters,
                                      kernel_size=4,
                                      padding="valid",
                                      activation="relu")
        self.pool = layers.GlobalMaxPool1D() # no training variable so we can
                                             # use the same layer for each
                                             # pooling step
        self.dense_1 = layers.Dense(units=FFN_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if nb_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=nb_classes,
                                           activation="softmax")
    
    def call(self, inputs, training):
        x = self.embedding(inputs)
        x_1 = self.bigram(x)
        x_1 = self.pool(x_1)
        x_2 = self.trigram(x)
        x_2 = self.pool(x_2)
        x_3 = self.fourgram(x)
        x_3 = self.pool(x_3)
        
        merged = tf.concat([x_1, x_2, x_3], axis=-1) # (batch_size, 3 * nb_filters)
        merged = self.dense_1(merged)
        merged = self.dropout(merged, training)
        output = self.last_dense(merged)
        
        return output

if __name__=='__main__':
    df=pd.read_csv(config.PROCESSED_FILEPATH)
    seq_text=preprocess.SeqFromTextConversionSK()
#     X=seq_text.fit_transform(df[config.PROC_TEXT_COL][:5].astype(str))
#     print(X)
    
