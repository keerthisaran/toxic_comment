import tensorflow as tf
from tensorflow.keras import layers


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class Toxic_Transformer(tf.keras.Model):
    
    def __init__(self,vocab_size,embed_dim, num_heads, ff_dim, rate, maxlen,nb_classes):
        super().__init__()
        self.multihead=TransformerBlock(embed_dim, num_heads, ff_dim, rate)
        self.pos_tok_embedding=TokenAndPositionEmbedding(maxlen,vocab_size,embed_dim)

        
        self.glob1=layers.GlobalAveragePooling1D()
        self.drop1=layers.Dropout(0.1)
        self.dense1=layers.Dense(20, activation="relu")
        self.drop1=layers.Dropout(0.1)


        if nb_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=nb_classes,
                                           activation="softmax")
    
    def call(self,inputs,training=True):
        # embedding_layer = 
        x = self.pos_tok_embedding(inputs)
        x = self.multihead(x)
        x=  self.glob1(x)
        x=  self.drop1(x,training)
        x=  self.dense1(x)
        x=  self.drop1(x)

        outputs = self.last_dense(x)
        return outputs

    def model(self):
        x = tf.keras.layers.Input(shape=(None,),name='Input')
        return tf.keras.Model(inputs=[x], outputs=self.call(x,True))

        
