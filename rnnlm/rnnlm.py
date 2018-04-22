import tensorflow as tf

class RNNLM(object):
    def __init__(self, config):
        '''
        :param config: a dict indicates the parameters for the model.
        '''
        self.config = config
        self.vocab_size = 12
        self.embed_size = 50
        self.sequence_size = 60

    def model_fn(self, features, labels, mode):
        """
        :param features: the source sequence. Tensor of shape [batch_size, sequence_length]
        :param labels: the target sequence. Tensor of shape [batch_size, sequence_length]
        :return: 
        """
        training = mode == tf.estimator.ModeKeys.TRAIN

        # embedding layer
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            embedding_table = tf.get_variable("embedding", [self.vocab_size, self.embed_size])
            inputs = tf.nn.embedding_lookup(embedding_table, features)

        # rnn layer
        tf.keras.layers.LSTM




