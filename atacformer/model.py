from kipoiseq.dataloaders import SeqIntervalDl
import pyranges

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from tensorflow.keras.layers import Embedding, Input, GlobalAveragePooling1D, Dense
from tensorflow.keras.layers import Conv1D, GlobalMaxPool1D

from tensorflow.keras.models import Sequential, Model
import numpy as np

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), 
             Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

#TODO:change
class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
   
    

def model_seq_to_ATACsignal(
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
):
    vocab_size = 4  # Letters [ACGT]
    maxlen = 2000 #auto-resize

    inputs = Input(shape=(maxlen,4))
    #embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    #x = embedding_layer(inputs)
    conv = Conv1D(32, 7)
    x = conv(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    #x = GlobalAveragePooling1D()(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(20, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(1)(x)
    #outputs = Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def model_seq_atac_to_gene_expression():
    pass


if __name__ == '__main__':
    model = model_seq_to_ATACsignal()
    model.compile('adam', loss="mse", metrics=["mse"])
    dl = SeqIntervalDl(fasta_file = 'mm10.fa', intervals_file = 'diff_motifs_homeobox.bed4.bed', auto_resize_len=2000)
    
    train = dl.load_all(batch_size=32)
    x_train = train['inputs']
    y_train = train['targets']
    
    history = model.fit(x_train, y_train, 
                    batch_size=64, epochs=20, 
                    #validation_data=(x_val, y_val)
                   )
