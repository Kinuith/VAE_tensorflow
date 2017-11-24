# coding = utf-8

import sys,os
import tensorflow as tf
import numpy as np

from utils import read_data_sets

class VariableAutoEncode(object): # Only for mnist



    def __init__(self,data):

        self.input_size = 784
        self.hidden_size = 500
        self.output_size = 30


        x = tf.placeholder(tf.float32,(None,784),'input')

        w_encode_1 = self.w_variable((self.input_size,self.hidden_size))
        b_encode_1=  self.b_variable((self.hidden_size,))

        h = tf.nn.relu(tf.matmul(x,w_encode_1) + b_encode_1)

        w_encode_2 = self.w_variable((self.hidden__size, self.output_size))
        b_encode_2 = self.b_variable((self.output_size,))

        o = tf.nn.relu(tf.matmul(h, w_encode_2) + b_encode_2)

        mn  = tf.reduce_mean(o,reduction_indices=1)
        sd =  tf.reduce_mean(o*o,reduction_indices=1) - tf.reduce_mean(mn*mn,reduction_indices=1)

        sd = tf.sqrt(sd)

# encoding finished !




    def w_variable(self,shape):
        w  = tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(w)

    def b_variable(self,shape):
        b = tf.zeros(shape)
        return tf.Variable(b)