# coding: utf-8
import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import Embedding,SigmoidWithLoss
import collections

class EmbeddingDot:
	def __init__(self,W):
		self.embed=Embedding(W)
		self.params=self.embed.params
		self.grads=self.embed.grads
		self.cache=None

	def forward(self,h,idx):
		target_W=self.embed.forward(idx)#重要
		out=np.sum(target_W*h,axis=1)
		
		self.cache=(h,target_W)
		return out

	def backward(self,dout):
		h,target_W=self.cache
		dout=dout.reshape(dout.shape[0],1)
		
		dtarget_W=dout*h
		self.embed.backward(dtarget_W)
		dh=dout*target_W
		return dh

class UnigramSampler:
	def __init__(self,corpus,power,sample_size):
		self.sample_size=sample_size
		self.vocab_size=None
		self.word_p=None
		
		counts=collections.Counter()
		for word_id in corpus:
			counts[word_id]+=1#重要

		vocab_size=len(counts)
		self.vocab_size=vocab_size

		self.word_p=np.zeros(vocab_size)
		for i in range(vocab_size):
			self.word_p[i]=counts[i]

		self.word_p=np.power(self.word_p,power)#累乗
		self.word_p/=np.sum(self.word_p)

	def get_negative_sample(self,target):
		batch_size = target.shape[0]

		if not GPU:
			negative_sample=np.zeros((batch_size,self.sample_size),dtype=np.int32)

			for i in range(batch_size):
				p=self.word_p.copy()#重要		
				target_idx=target[i]
				p[target_idx]=0
				p/=p.sum()
				negative_sample[i,:]=np.random.choice(self.vocab_size,size=self.sample_size,replace=False,p=p)
		else:
			#GPU(cupy)利用時には速度優先
			#負例にターゲットが含まれるケースあり
			negative_sample=np.random.choice(self.vocab_size,size=(batch_size,self.sample_size),replace=True,p=self.word_p)
		return negative_sample

class NegativeSamplingLoss:
	def __init__(self,W,corpus,power=0.75,sample_size=5):
		self.sample_size=sample_size
		self.sampler=UnigramSampler(corpus,power,sample_size)
		self.loss_layers=[SigmoidWithLoss() for _ in range(sample_size+1)]
		self.embed_dot_layers=[EmbeddingDot(W) for _ in range(sample_size+1)]
		
		self.params,self.grads=[],[]
		for layer in self.embed_dot_layers:
			self.params += layer.params
			self.grads += layer.grads


	def forward(self,h,target):
		batch_size=target.shape[0]
		negative_sample=self.sampler.get_negative_sample(target)

		#正例のフォワード
		score=self.embed_dot_layers[0].forward(h,target)
		correct_label=np.ones(batch_size,dtype=np.int32)
		loss=self.loss_layers[0].forward(score,correct_label)

		#負例のフォワード
		negative_label=np.zeros(batch_size,dtype=np.int32)#修正箇所(×dtype=int32)
		for i in range(self.sample_size):
			negative_target=negative_sample[:,i]
			score=self.embed_dot_layers[1+i].forward(h,negative_target)
			loss+=self.loss_layers[1+i].forward(score,negative_label)

		return loss

	def backward(self,dout=1):
		dh=0
		for l0,l1 in zip(self.loss_layers,self.embed_dot_layers):
			dscore=l0.backward(dout)
			dh+=l1.backward(dscore)
		return dh































