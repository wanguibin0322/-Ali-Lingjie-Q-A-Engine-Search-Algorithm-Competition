import tensorflow as tf
import os
import argparse
import numpy as np
import time

from bert import modeling

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


class RankModel(object):
    def __init__(self, query_input_ids, doc_input_ids, max_seq_length):

        query_input_shape = get_shape_list(query_input_ids)  # the shape of query_input_ids may be (1, x)
        query_seq_length = query_input_shape[1]

        doc_input_shape = get_shape_list(doc_input_ids) 
        doc_batch_size = doc_input_shape[0] 

        # first finds nonzeros elements, then takes these nonzero-elements' indices, finally expands dims
        nonzero_indices = tf.where(tf.greater(query_input_ids, 0))  # dim = (num_nonzero, input_ids.rank)
        nonzero_indices_shape = get_shape_list(nonzero_indices)

        q_input_ids = tf.tile(tf.expand_dims(tf.gather_nd(query_input_ids, nonzero_indices), 0), [doc_batch_size, 1])
        
        self.input_ids = tf.slice(tf.concat([q_input_ids, doc_input_ids], axis=-1), [0,0], [-1, max_seq_length])
        self.create_model()

    def create_model(self):
        """ This part is almost the same as your model for training, but removing ** label informations **.
        """
        with tf.variable_scope("embeddings"):
            embedding_output, _ = embedding_lookup(self.input_ids)
        with tf.variable_scope("encoder"):
            reduce_output = tf.reduce_sum(embedding_output , axis=1)
            logits = tf.layers.dense(reduce_output, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            prob = tf.sigmoid(logits)
        self.score = tf.identity(prob, name="score")  # note: don't define self.score in tf.variable_scope

class WrapperModel(object):
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length

        self.query_ids = tf.placeholder(dtype=tf.int32, shape=[None, 128], name="query_ids")
        self.doc_ids = tf.placeholder(dtype=tf.int32, shape=[None, 128], name="doc_ids")

        self.create_model()

    def create_model(self):
        model = RankModel(query_input_ids=self.query_ids, doc_input_ids=self.doc_ids, max_seq_length=self.max_seq_length)
        self.score = model.score 

def save_to_pb(sess, model, pb_model_path):
    builder = tf.saved_model.builder.SavedModelBuilder(pb_model_path)
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
                       inputs={"query_ids": model.query_ids, "doc_ids": model.doc_ids},
                       outputs={"score": model.score})
    builder.add_meta_graph_and_variables(sess=sess,
                                 tags=[tf.saved_model.tag_constants.SERVING],
                                 signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
    builder.save()
    print("saved to ", pb_model_path)

def test_saved_model(saved_model_path):
    """
    The input_ids of query and doc are fake.
    Note:
        If you use bert-like model, 
        *** don't forget add the index of '[CLS]' and '[SEP]' in the input_ids of query. 
        *** don't forget add the index of '[SEP]' in the input_ids of doc.
    """
    query_input_ids = np.array([[101, 4508, 7942, 7000, 7350, 2586, 3296, 2225, 4275, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).reshape(1, 128)
    doc_input_ids = np.array([[2408, 691, 7556, 705, 4385, 6573, 6862, 1355, 524, 11361, 120, 2608, 4448, 5687, 1788, 4508, 4841, 7000, 7350, 2364, 3296, 2225, 4275, 121, 119, 8132, 8181, 115, 8108, 4275, 120, 4665, 166, 8197, 5517, 7608, 5052, 5593, 4617, 3633, 1501, 3241, 3309, 5310, 1394, 6956, 5593, 4617, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[727, 1310, 4377, 4508, 4841, 7000, 796, 827, 3296, 2225, 5540, 1718, 9695, 8181, 115, 8114, 5108, 120, 4665, 5498, 5301, 5528, 4617, 1146, 1265, 1798, 4508, 4307, 5593, 4617, 2229, 6956, 3241, 3309, 3186, 5664, 2421, 2135, 3175, 3633, 1501, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[928, 6449, 3710, 4508, 4384, 7000, 4275, 8114, 3710, 7185, 2128, 4508, 4384, 7000, 5790, 4508, 4841, 7000, 1079, 3302, 1366, 3302, 928, 2139, 7212, 1853, 3632, 6117, 5542, 4508, 7000, 1980, 6612, 3714, 4508, 3705, 3186, 5664, 2421, 3130, 1744, 772, 3709, 5790, 4275, 6820, 677, 3862, 4508, 3710, 3710, 7000, 5540, 1718, 5162, 7942, 7000, 3714, 2135, 3175, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).reshape(-1,128)

    with tf.Session(graph=tf.Graph()) as sess:
        loaded = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_path)
        graph = tf.get_default_graph()
        query = sess.graph.get_tensor_by_name('query_ids:0')
        doc = sess.graph.get_tensor_by_name('doc_ids:0')
        score = sess.graph.get_tensor_by_name('score:0')
        starttime = time.time()
        score = sess.run(score, feed_dict={query:query_input_ids, doc:doc_input_ids})
        print("score: ", score)
        print("*** Please check if this score is correct.")
    print("*** The saved_model is available.")


def run_save_model(args):
    print("*** Converting model ...")
    ckpt_to_convert = args.ckpt_to_convert
    model = WrapperModel(max_seq_length=args.max_seq_length) 
    with tf.Session() as sess:
        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
                        tvars, ckpt_to_convert)
        tf.train.init_from_checkpoint(ckpt_to_convert, assignment_map)
        sess.run(tf.variables_initializer(tf.global_variables()))
        save_to_pb(sess, model, args.output_dir)
    print("*** Saving model to 【%s】 ***" % args.output_dir)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_to_convert", type=str, help="the checkpoint of trained model to convert") 
    parser.add_argument("--output_dir", type=str, help="the path of saved_model") 
    parser.add_argument("--max_seq_length", type=int)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    
    run_save_model(args)
    test_saved_model(args.output_dir)
