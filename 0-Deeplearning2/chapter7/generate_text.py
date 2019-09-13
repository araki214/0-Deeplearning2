import sys
sys.path.append('..')
from rnnlm_gen import RnnlmGen
from dataset import ptb

corpus,wordtoid,idtoword=ptb.load_data('train')
vocab_size=len(wordtoid)
corpus_size=len(corpus)

model=RnnlmGen()
model.load_params('../chapter6/BetterRnnlm.pkl')

#start文字とskip文字の設定
start_word='you'
start_id=wordtoid[start_word]
skip_words=['N','<unk>','$']
skip_ids=[wordtoid[i] for i in skip_words]

#文章生成
word_ids=model.generate(start_id,skip_ids)
txt=' '.join([idtoword[i] for i in word_ids])
txt=txt.replace(' <eos>','.\n')
print(txt)
