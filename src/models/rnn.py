from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras import layers



class RNN(tf.keras.Model):
    
    def __init__(self,
                 vocab_size,
                 emb_dim=128,
                 gru_units=64,
                 FFN_units=512,
                 nb_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="dcnn"):
        super().__init__(name=name)
        
        self.embedding = layers.Embedding(vocab_size,
                                          emb_dim)
        self.gru=tf.keras.layers.Bidirectional(
                    layer=tf.keras.layers.GRU(units=gru_units))

        self.dense_1 = layers.Dense(units=FFN_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)

        if nb_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=nb_classes,
                                           activation="softmax")
    
    def call(self, inputs, training=True):
        
        x = self.embedding(inputs)
        gru_last_hidden=self.gru(x)
        
        merged = self.dense_1(gru_last_hidden)
        merged = self.dropout(merged, training)
        output = self.last_dense(merged)
        
        return output
      
    def model(self):
        #for model summary this function initializes the graph
        #DCNN(vocab_size=1000).model().summary()
      x = tf.keras.layers.Input(shape=(None,),name='Input')
      return tf.keras.Model(inputs=[x], outputs=self.call(x,True))
    
    
    
    
    
