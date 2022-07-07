import tensorflow as tf
import os
import argparse
import numpy as np


class Ranker(object):
    """This model is just for training."""
    def __init__(self, is_training=False, max_seq_length=100): 
        self.query_ids = tf.placeholder(dtype=tf.int32, shape=[1, None], name='query_ids')
        self.doc_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='doc_ids')
        self.max_seq_length = max_seq_length

        self.create_model()
        self.init_saver()

    def create_model(self):
        batch_size = tf.shape(self.doc_ids)[0]
        expanded_query_ids = tf.tile(self.query_ids, [batch_size, 1])
        input_ids = tf.slice(tf.concat([expanded_query_ids, self.doc_ids], axis=-1), [0,0], [-1, self.max_seq_length])
        with tf.variable_scope("embeddings"):
            embedding_output, _ = embedding_lookup(input_ids)
        with tf.variable_scope("encoder"):
            reduce_output = tf.reduce_sum(embedding_output , axis=1)
            logits = tf.layers.dense(reduce_output, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            prob = tf.sigmoid(logits)
        self.score = tf.identity(prob, name="score")  # note: don't define self.score in tf.variable_scope 

    def init_saver(self):
         self.saver = tf.train.Saver(tf.global_variables())


def embedding_lookup(input_ids,
                       vocab_size=21128,
                       embedding_size=128,
                       initializer_range=0.02,
                       word_embedding_name="word_embeddings",
                       use_one_hot_embeddings=False):
    if input_ids.shape.ndims == 2:
      input_ids = tf.expand_dims(input_ids, axis=[-1])
  
    embedding_table = tf.get_variable(
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
  
    flat_input_ids = tf.reshape(input_ids, [-1])
    if use_one_hot_embeddings:
      one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
      output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
      output = tf.gather(embedding_table, flat_input_ids)
  
    input_shape = get_shape_list(input_ids)
  
    output = tf.reshape(output,
                        input_shape[0:-1] + [input_shape[-1] * embedding_size])
    return (output, embedding_table)


def get_shape_list(tensor, expected_rank=None, name=None):
    if name is None:
      name = tensor.name
  
    if expected_rank is not None:
      assert_rank(tensor, expected_rank, name)
  
    shape = tensor.shape.as_list()
  
    non_static_indexes = []
    for (index, dim) in enumerate(shape):
      if dim is None:
        non_static_indexes.append(index)
  
    if not non_static_indexes:
      return shape
  
    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
      shape[index] = dyn_shape[index]
    return shape



if __name__ == "__main__":
    save_path = "temp"
    model = Ranker()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        model.saver.save(sess, save_path, global_step=0)

