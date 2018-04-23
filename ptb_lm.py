'''
rnn language model
'''

import os
cwd = os.path.split(os.path.realpath(__file__))[0]
data_dir = os.path.join(cwd, "corpora", "ptb")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

from dataset import ptb

ptb_data = ptb.PTB(data_dir)

args = {
  "vocab_size" : len(ptb_data.word2id),
  "batch_size" : 32,
  "sequence_size" : 35,
  "num_units" : 1500,
  "embed_size" : 1500
}

from rnnlm import rnnlm
from rnnlm import input_fn
model = rnnlm.RNNLM(**args)

import tensorflow as tf
estimator = tf.estimator.Estimator(
  model_fn=model.model_fn,
  model_dir=os.path.join(data_dir, "model")
)

estimator.train(
  input_fn=lambda : input_fn.input_fn(ptb_data.train, args["sequence_size"], args["batch_size"]),
  steps = 5000
)
