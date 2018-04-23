import tensorflow as tf

class RNNLM(object):
    def __init__(self, vocab_size, embed_size, sequence_size, num_units, batch_size):
        '''
        :param config: a dict indicates the parameters for the model.
        '''
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.sequence_size = sequence_size
        self.num_units = num_units
        self.batch_size = batch_size
        self.dropout_keep_prob = 0.5
        self.num_layers = 2
        self.lr = 0.001

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
        # Shape of outputs : [batch_size, sequence_size, num_units]
        def make_cell():
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.num_units, forget_bias=1.0, state_is_tuple=True)
            if training and self.dropout_keep_prob < 1.0:
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)
            return lstm_cell
        lstm_cell = tf.nn.rnn_cell.MultiRNNCell([make_cell() for _ in range(self.num_layers)], state_is_tuple=True)
        init_sate = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=inputs, initial_state=init_sate, dtype=tf.float32)
        outputs = tf.reshape(outputs, [-1, self.num_units])

        # loss
        softmax_w = tf.get_variable("softmax_w", [self.num_units, self.vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [self.vocab_size], dtype=tf.float32)
        logits = tf.nn.xw_plus_b(outputs, softmax_w, softmax_b)
        logits = tf.reshape(logits, [self.batch_size, self.sequence_size, self.vocab_size])

        predictions = {
            "setences" : tf.argmax(logits, axis=2)
        }
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            labels,
            tf.ones([self.batch_size, self.sequence_size]),
            average_across_timesteps=False,
            average_across_batch=True
        )
        loss = tf.reduce_sum(loss)

        if training:
            train_op = tf.train.AdamOptimizer(self.lr).minimize(loss=loss,global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)




