import tensorflow as tf


def input_fn(word_ids, sequence_len, batch_size, buffer_size=500, training = True):
    sample_num = (word_ids.size - 1) // sequence_len
    x = word_ids[:sample_num*sequence_len].reshape(sample_num, sequence_len)
    y = word_ids[1:(sample_num*sequence_len+1)].reshape(sample_num, sequence_len)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        dataset = dataset.shuffle(buffer_size).repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()





