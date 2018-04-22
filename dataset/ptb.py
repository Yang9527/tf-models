import tensorflow as tf
from os.path import join
import numpy as np


class PTB(object):
    def __init__(self, dirname):
        self.__filename = join(dirname, "simple-examples.tgz")
        if not tf.gfile.Exists(self.__filename):
            from .util import download
            url = "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"
            download(url, self.__filename)
        self.word2id = dict()
        self.__build_vocab()
        self.train = np.array(self.__index(self.__read_train()), dtype=np.int32)
        self.test = np.array(self.__index(self.__read_test()), dtype=np.int32)
        self.valid = np.array(self.__index(self.__read_valid()), dtype=np.int32)

    def __read(self, _filename):
        import tarfile
        from contextlib import closing
        with closing(tarfile.open(self.__filename, 'r')) as t:
            f = t.extractfile(_filename)
            return f.read().decode("utf-8")

    def __read_train(self):
        return self.__read("./simple-examples/data/ptb.train.txt")

    def __read_test(self):
        return self.__read("./simple-examples/data/ptb.test.txt")

    def __read_valid(self):
        return self.__read("./simple-examples/data/ptb.valid.txt")

    def __build_vocab(self):
        words = self.__read_train().replace("\n", ' <eos> ').split()
        from collections import Counter
        cnt = Counter(words)
        cnt_pairs = sorted(cnt.items(), key=lambda x : (-x[1], x[0]))
        self.word2id = dict([(word, i) for i, (word, _) in enumerate(cnt_pairs)])

    def __index(self, string):
        words = string.replace("\n", " <eos> ").split()
        return [self.word2id[word] for word in words]
