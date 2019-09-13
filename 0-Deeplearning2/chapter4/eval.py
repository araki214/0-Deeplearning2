import sys 
sys.path.append('..')
from common.util import most_similar,analogy
import pickle

pkl_file='cbow_params.pkl'
with open(pkl_file, 'rb') as f:
	params=pickle.load(f)
	word_vecs=params['word_vecs']
	wordtoid=params['wordtoid']
	idtoword=params['idtoword']


querys=['you','year','car','toyota']
for query in querys:
	most_similar(query,wordtoid,idtoword,word_vecs,top=5)

analogy('king','man','queen',wordtoid,idtoword,word_vecs)
analogy('take','took','go',wordtoid,idtoword,word_vecs)
analogy('car','cars','child',wordtoid,idtoword,word_vecs)
analogy('good','better','bad',wordtoid,idtoword,word_vecs)

