from src.models.cnn import DCNN
from src.models.sklearn_wrapper import TF_to_SK

from preprocess.seq_from_text import SeqFromTextConversionSK
import config
import pandas as pd
import tensorflow as tf
import numpy as np

df=pd.read_csv(config.PROCESSED_FILEPATH)
df[config.PROC_TEXT_COL]
seq_text=SeqFromTextConversionSK()
X=seq_text.fit_transform(df[config.PROC_TEXT_COL].astype(str))

compile_params=dict(               
               loss=tf.keras.losses.BinaryCrossentropy(),
               optimizer=tf.keras.optimizers.Adam(),
               metrics=[tf.keras.metrics.BinaryAccuracy()],
)
init_params=dict(    
                 vocab_size=len(seq_text.tokenizer.vocabulary_)+2,
                 emb_dim=128,
                 nb_filters=50,
                 FFN_units=512,
                 nb_classes=2)

model=TF_to_SK(DCNN,init_params,compile_params,batch_size=128,epochs=2)
X=np.array(X)
y=df[config.LABEL].values
model.fit(X,y)
model.predict(X)

