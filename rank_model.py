import tensorflow as tf
import os
import argparse
import numpy as np

from bert import modeling
from bert import optimization


class Ranker(object):
    """This model is just for training."""
    def __init__(self, bert_config_path, is_training, num_train_steps=None, num_warmup_steps=None, learning_rate=1e-5):
        self.bert_config_path = bert_config_path
        self.learning_rate = learning_rate
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.is_training = is_training

        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_ids")
        self.input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_mask")
        self.token_type_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="token_type_ids")
        self.label_ids = tf.placeholder(dtype=tf.int32, shape=[None,], name="label_ids")
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=None, name="batch_size")
        
        self.create_model()
        self.init_saver()

    def create_model(self):
        num_labels = 2
        self.bert_config = modeling.BertConfig.from_json_file(self.bert_config_path)
        model = modeling.BertModel(config=self.bert_config, is_training=self.is_training, 
                                   input_ids=self.input_ids, input_mask=self.input_mask, token_type_ids=self.token_type_ids,
                                   use_one_hot_embeddings=False)
        output_layer = model.get_pooled_output()
        self.output_layer = output_layer
        hidden_size = output_layer.shape[-1].value

        with tf.variable_scope("projection", reuse=tf.AUTO_REUSE):
            output_weights = tf.get_variable("output_weights", [num_labels-1, hidden_size], initializer=tf.truncated_normal_initializer(stddev=0.02)) # num_label=2
            output_bias = tf.get_variable("output_bias", [num_labels-1], initializer=tf.zeros_initializer())
    
        with tf.variable_scope("loss"):
            if self.is_training:
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            self.logits = tf.nn.bias_add(logits, output_bias)
            self.probabilities = tf.nn.sigmoid(self.logits)
            self.score = tf.identity(self.probabilities, name="score")

    
        if self.is_training:
            with tf.name_scope("train_op"):
                self.label_ids = tf.cast(self.label_ids, dtype=tf.float32)
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.reshape(self.label_ids, (-1, self.logits.shape[-1].value))))
                '''
                alpha = 0.25
                gamma = 2
                pred = tf.nn.sigmoid(self.logits)
                y = tf.reshape(self.label_ids, (-1, self.logits.shape[-1].value))
                #print("y size: ",tr.size(y))
                zeros = tf.zeros_like(pred, dtype=pred.dtype)
                #print("zeros size:",tf.size(zeros))
                pos_p_sub = tf.where(y > zeros, y - pred, zeros) # positive sample 寻找正样本，并进行填充
                neg_p_sub = tf.where(y > zeros, zeros, pred) # negative sample 寻找负样本，并进行填充
                per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(pred, 1e-8, 1.0)) \
                 	             - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - pred, 1e-8, 1.0))
                self.loss = tf.reduce_sum(per_entry_cross_ent)
                '''
                self.train_op = optimization.create_optimizer(
                    self.loss, self.learning_rate, self.num_train_steps, self.num_warmup_steps, use_tpu=False)

    def init_saver(self):
        self.saver = tf.train.Saver(tf.global_variables())

   
    def train(self, sess, batch):
        batch_size = len(batch["input_ids"])
        feed_dict = {self.input_ids: batch["input_ids"],
                     self.input_mask: batch["input_mask"],
                     self.token_type_ids: batch["token_type_ids"],
                     self.label_ids: batch["label_ids"],
                     self.batch_size: batch_size}
        _, loss,logits, prob = sess.run([self.train_op, self.loss, self.logits, self.probabilities], feed_dict=feed_dict)
        return loss, logits, prob

def focal_loss(pred, y, alpha=0.25, gamma=2):
        r"""Compute focal loss for predictions.
            Multi-labels Focal loss formula:
                FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                     ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
        Args:
         pred: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
         y: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
         alpha: A scalar tensor for focal loss alpha hyper-parameter
         gamma: A scalar tensor for focal loss gamma hyper-parameter
        Returns:
            loss: A (scalar) tensor representing the value of the loss function
        """

        zeros = tf.zeros_like(pred, dtype=pred.dtype)
        pos_p_sub = tf.where(y > zeros, y - pred, zeros) # positive sample 寻找正样本，并进行填充
        # For positive prediction, only need consider front part loss, back part is 0;
        # target_tensor > zeros <=> z=1, so positive coefficient = z - p.

        neg_p_sub = tf.where(y > zeros, zeros, pred) # negative sample 寻找负样本，并进行填充
        # For negative prediction, only need consider back part loss, front part is 0;
        # target_tensor > zeros <=> z=1, so negative coefficient = 0.
        per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(pred, 1e-8, 1.0)) \
                              - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - pred, 1e-8, 1.0))
        return tf.reduce_sum(per_entry_cross_ent)

def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss
