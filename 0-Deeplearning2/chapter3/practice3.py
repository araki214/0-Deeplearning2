import numpy as np
c=np.array([1,0,0,0,0,0,0])
W=np.random.randn(7,3)#重要
h=np.dot(c,W)

import sys
sys.path.append('..')
from common.util import preprocess,create_contexts_target,convert_one_hot

text='You say goodbye and i say hello.'
corpus,wordtoid,idtoword=preprocess(text)
print(corpus)

print(idtoword)

#ターゲットとコンテキストをそれぞれ生成
contexts,target=create_contexts_target(corpus,window_size=1)

print(contexts)
print(target)

#one-hotベクトルに
vocab_size=len(wordtoid)
target=convert_one_hot(target,vocab_size)
contexts=convert_one_hot(contexts, vocab_size)

print(contexts)
print(target)


