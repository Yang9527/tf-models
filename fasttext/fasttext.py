import tensorflow as tf


class FastText(object):
    def __init__(self, config):
        """
        :param config: configuration of the model. must have the following fields:
            embeddings -- a numpy vector specialize the initial embedding lookup table
            vocab_size -- vocabulary size of the embedding lookup table
            embed_size -- the dimension of the embedding
            loss_type -- nce or softmax
            
            
            
        """
        self.config = config


    def model_fn(self, features, labels, mode):
        """
        :param features: [batch_size, sequence_length]
        :param labels: [batch_size]
        :param mode: TRAIN | EVALUATE | PREDICT
        :return: Estimator
        """
        # embedding layer
        with tf.device("/cpu:0"), tf.name_scope("input"):
            embedding_table = tf.get_variable("embedding",
                                              [self.config.vocab_size, self.config.embed_size],
                                              initializer=self.config.embeddings)
            inputs = tf.nn.embedding_lookup(embedding_table, features)

        logits = tf.reduce_mean(inputs, axis=1)
        logits = tf.layers.dense(inputs=logits, units=self.config.n_class, activation=tf.nn.relu)

        predictions = {
            "classes": tf.argmax(logits, axis=1),
            "probabilities": tf.nn.softmax(logits)
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        tf.summary.scalar("loss", loss)

        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
        }
        tf.summary.scalar("accuracy", eval_metric_ops["accuracy"][1])

        summary_hook = tf.train.SummarySaverHook(
            self.config.save_summary_steps,
            self.config.save_summary_dir
        )

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[summary_hook])

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
