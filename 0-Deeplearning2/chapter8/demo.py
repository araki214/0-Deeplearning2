import numpy as np

N,T,H=10,5,4
hs=np.random.randn(N,T,H)
a=np.random.randn(N,T)

ar=a.reshape(N,T,1).repeat(H,axis=2)
print(ar.shape)
t=hs*ar
print(t.shape)
c=np.sum(t,axis=1)
print(c.shape)

