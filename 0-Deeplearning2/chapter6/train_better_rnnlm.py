import sys 
sys.path.append('..')
from common import config
#GPU
config.GPU=True
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity
from dataset import ptb
from better_rnnlm import BetterRnnlm

#ハイパーパラメータの設定
batch_size=20
wordvec_size=650
hidden_size=650
time_size=35
lr=20.0
max_epoch=40
max_grad=0.25
dropput=0.5

#学習データの読み込み
corpus,wordtoid,idtoword=ptb.load_data('train')
corpus_val,_,_=ptb.load_data('val')
corpus_test,_,_=ptb.load_data('test')

vocab_size=len(wordtoid)
#print(corpus)
xs=corpus[:-1]
ts=corpus[1:]
#print(xs)
#print(ts)

model=BetterRnnlm(vocab_size,wordvec_size,hidden_size,dropput)
optimizer=SGD(lr)
trainer=RnnlmTrainer(model,optimizer)

best_ppl=float('inf')
for epoch in range(max_epoch):
	trainer.fit(xs,ts,max_epoch=1,batch_size=batch_size,time_size=time_size,max_grad=max_grad)
	model.reset_state()
	ppl=eval_perplexity(model,corpus_val)
	print('valid perplexity:',ppl)

	if best_ppl>ppl:
		best_ppl=ppl
		model.save_params()
	else:
		lr/=4.0
		optimizer.lr=lr
	model.reset_state()
	print('-'*50)

trainer.plot(ylim=(0,500))
model.reset_state()
ppl_test=eval_perplexity(model,corpus_test)
print('test perplexity:',ppl_test)

