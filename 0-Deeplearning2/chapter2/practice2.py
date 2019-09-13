import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess,create_to_matrix,cos_similarity,most_similar,ppmi
from dataset import ptb
text='You say goodbye and I say hello.'
corpus,wordtoid,idtoword=preprocess(text)

#手作り
C=np.array([
[0,1,0,0,0,0,0],
[1,0,1,0,1,1,0],
[0,1,0,1,0,0,0],
[0,0,1,0,1,0,0],
[0,1,0,1,0,0,0],
[0,1,0,0,0,0,1],
[0,0,0,0,0,1,0],
],dtype=np.int32)
print(C[0])
print(C[4])
print(C[wordtoid['goodbye']])

vocab_size=len(wordtoid)
C=create_to_matrix(corpus,vocab_size,window_size=1)

#similarity
c0=C[wordtoid['you']]
c1=C[wordtoid['i']]
print(cos_similarity(c0,c1))

#most_similar
most_similar('you',wordtoid,idtoword,C,top=5)

#ppmi
W=ppmi(C)

np.set_printoptions(precision=3)
print('covariance matrix')
print(C)
print('-'*50)
print('PPMI')
print(W)

#count_method_small
U,S,V=np.linalg.svd(W)
print(C[0])
print(W[0])
print(U[0])
print(U[0, :2])

for word, word_id in wordtoid.items():
	plt.annotate(word,(U[word_id,0],U[word_id,1]))

plt.scatter(U[:,0],U[:,1],alpha=0.5)
plt.show()

#show_ptb
corpus, word_to_id,id_to_word=ptb.load_data('train')
print('corpus size:',len(corpus))
print('corpus[:30]:',corpus[:30])
print()
print('id_to_word[0]:',id_to_word[0])
print('id_to_word[1]:',id_to_word[1])
print('id_to_word[2]:',id_to_word[2])
print()
print("word_to_id['car']:",word_to_id['car'])
print("word_to_id['happy']:",word_to_id['happy'])
print("word_to_id['lexus']:",word_to_id['lexus'])



