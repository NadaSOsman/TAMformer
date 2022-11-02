import os
import sys
import yaml
import numpy as np
import tensorflow as tf
import random as rn
import copy
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Concatenate, BatchNormalization, Softmax, Flatten, Add, Activation
from tensorflow.keras import layers, activations
from tensorflow import keras


class masking_models(layers.Layer):
    def __init__(self, final_out=1, func='sigmoid'):
        super(masking_models, self).__init__()
        self.masking_model = keras.Sequential([Dense(128, activation='relu'),
                                               Dropout(0.1),
                                               Dense(64, activation='relu'),
                                               Dropout(0.1),
                                               Dense(32, activation='relu'),
                                               Dropout(0.1),
                                               Dense(final_out, activation=func)])

    def call(self, inputs):
        return self.masking_model(inputs)



class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, normalization=True, cross_attention=False):
        super(TransformerBlock, self).__init__()
        self.cross_attention = cross_attention
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        if normalization:
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.normalization = normalization

    def call(self, inputs, training, attention_mask=None):
        if self.cross_attention:
            if not training:
                attention_mask = tf.round(attention_mask)
            attn_output = self.att(inputs[0], inputs[1], attention_mask=attention_mask)
            attn_output = self.dropout1(attn_output, training=training)
            if self.normalization:
                out1 = self.layernorm1(inputs[0] + attn_output)
            else:
                out1 = inputs[0] + attn_output
        else:
            if not training:
                attention_mask = tf.round(attention_mask)
            attn_output = self.att(inputs, inputs, attention_mask=attention_mask)
            attn_output = self.dropout1(attn_output, training=training)
            if self.normalization:
                out1 = self.layernorm1(inputs + attn_output)
            else:
                out1 = inputs + attn_output
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        if self.normalization:
            return self.layernorm2(out1 + ffn_output)
        else:
            return out1 + ffn_output


class QueryEmbedding(layers.Layer):
    def __init__(self, num_of_queries, embed_dim):
        super(QueryEmbedding, self).__init__()
        self.query_emb = layers.Embedding(input_dim=num_of_queries, output_dim=embed_dim)
        self.num_of_queries = num_of_queries

    def call(self, x):
        queries = tf.range(start=0, limit=self.num_of_queries, delta=1)
        queries = self.query_emb(queries)
        return queries


class PositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions


class TAMformer(object):
    def __init__(self, model_opts=None, auxiliary_loss=False):
        self.model_opts = model_opts
        self.auxiliary_loss = auxiliary_loss


    def tamformer(self):
        num_modalities = len(self.model_opts['obs_input_type'])
        feat_sizes =  self.model_opts['feat_size']
        inputs = [Input((136, feat_sizes[i])) for i in range(num_modalities)]
        embeddings = [PositionEmbedding(136, feat_sizes[i])(inputs[i]) for i in range(num_modalities)]
        sampled_embd = [Lambda(lambda s: s[:,1::self.model_opts['step']])(embeddings[i]) for i in range(num_modalities)]
        concatenated_inputs = Concatenate(axis=-1)(inputs)
        sampled_concatenated_inputs = Lambda(lambda s: s[:,self.model_opts['obs_length']::self.model_opts['step']])(concatenated_inputs)


        masking_models_136 = [masking_models(final_out=i) for i in range(1,137)]
        masking_models_40 = [masking_models(final_out=i) for i in range(16,136,3)]

        masks_136 = [tf.expand_dims(masking_models_136[i](Lambda(lambda s, i=i: s[:,i])(concatenated_inputs)), 1) for i in range(136)]
        masks_136 = [tf.keras.layers.ZeroPadding1D((0, 136-masks_136[i].shape[-1]))(tf.transpose(masks_136[i], [0,2,1])) for i in range(136)]
        masks_136 = [tf.transpose(masks_136[i], [0,2,1]) for i in range(136)]
        masks_136 = Concatenate(axis=1)(masks_136)

        masks_40 = [tf.expand_dims(masking_models_40[i](Lambda(lambda s, i=i: s[:,i])(sampled_concatenated_inputs)), 1) for i in range(40)]
        masks_40 = [tf.keras.layers.ZeroPadding1D((0, 136-masks_40[i].shape[-1]))(tf.transpose(masks_40[i], [0,2,1])) for i in range(40)]
        masks_40 = [tf.transpose(masks_40[i], [0,2,1]) for i in range(40)]
        masks_40 = Concatenate(axis=1)(masks_40)

        transformer_blocks = [TransformerBlock(feat_sizes[i], 6, 1024, normalization=True, cross_attention=False)\
                                              (embeddings[i],attention_mask=masks_136) for i in range(num_modalities)]

        concatenated_encodings = Concatenate(axis=-1)(transformer_blocks)
        query_transformer = TransformerBlock(sum(feat_sizes), 6, 1024, normalization=True, cross_attention=True)\
                                            ([sampled_concatenated_inputs, concatenated_inputs], attention_mask=masks_40)

        cross_transformer_block = TransformerBlock(sum(feat_sizes), 6, 1024, normalization=True, cross_attention=True)\
                                                  ([query_transformer, concatenated_encodings], attention_mask=masks_40)

        outputs = []
        for i in range(40):
            x1 = Lambda(lambda s, i=i: s[:,i])(cross_transformer_block)
            x2 = Dropout(0.1)(x1)
            x3 = Dense(64, activation='relu')(x2)
            x4 = Dropout(0.1)(x3)
            x5 = Dense(32, activation='relu')(x4)
            x6 = Dropout(0.1)(x5)
            x7 = Dense(1, activation='sigmoid', name='output_'+str(i))(x6)
            outputs.append(x7)

        model = Model(inputs, outputs, name='tamformer')

        if self.auxiliary_loss:
            to_add_losses = []
            for i in range(40):
                #ce = K.binary_crossentropy(tf.round(o), tf.round(outputs[-1])) #L_r could be the ce with the closest prediction
                mse = K.square(Lambda(lambda s, i=i: s[:,i])(cross_transformer_block) - Lambda(lambda s, i=i: s[:,-1])(cross_transformer_block))
                loss = K.mean(mse)
                to_add_losses.append(loss)
            model.add_loss(to_add_losses)

        return model
