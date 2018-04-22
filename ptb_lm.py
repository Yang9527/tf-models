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

for k, v in list(ptb_data.word2id.items())[:10]:
    print(k, v)

print(ptb_data.train[:100])
print(ptb_data.test[:100])
print(ptb_data.valid[:100])
