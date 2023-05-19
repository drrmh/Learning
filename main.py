import torch
import numpy as np
import math
from matplotlib import pyplot as plt


# seed the pseudorandom number generator
from random import seed
from random import random
from d2l import torch as d2l

def curve(v,r,c,t):
    s1= v*c
    s2 = 1.8-math.exp(-1*t/(r*c))
    s = s1*s2
    vr=0
    if t>0 and t<250:
        vr=random()/2
    if t>250 and t<400:
        vr=random()/5
    if t>400 and t<600:
        vr=random()/8
    if t > 600 and t < 780:
         vr = random() / 10
    if t>780:
        vr= random()/50
        if t > 800:
            vr = random() / 80
        if t > 980:
             vr = 0

    return s-vr
def curve2(v,r,c,t):
    s=0
    CNST=1.8
    vr=0
    if t>0 and t<200:
        vr=random()/2
        s1 = 2.5 * c
        s2 = CNST - math.exp(-1 * t / (r * c))
        s = s1 * s2
    if t>200 and t<450:
        vr=random()/5
        s1 = 2.9 * c
        s2 = CNST - math.exp(-1 * t / (r * c))
        s = s1 * s2
    if t>450 and t<650:
        vr=random()/8
        s1 = 3.2 * c
        s2 = CNST - math.exp(-1 * t / (r * c))
        s = s1 * s2
    if t > 650 and t < 900:
        vr = random() / 10
        s1 = 3.92 * c
        s2 = CNST - math.exp(-1 * t / (r * c))
        s = s1 * s2
    if t>900:
        vr= random()/50
        s1 = 4 * c
        s2 = CNST - math.exp(-1 * t / (r * c))
        s = s1 * s2

        if t>990:
            vr=0
    rt = s-vr
    if rt<0:
        rt=rt*-1
    return rt



t =np.arange(5,1500,7)
x =np.arange(0,1500,7)
r1=1400
v1=5.8
c1=0.12

r=1500
v=2.4
c=0.2


plt.plot(t,[curve(v,r,c,t) for t in t], label = "Clus_Acc")
plt.plot(x,[curve2(v1,r1,c1,x) for x in x], label  = "Clas_ACC")
plt.ylim(0,1)
plt.xlim(0,1500)
plt.xlabel("Epochs")
plt.ylabel("ACC")
plt.legend()
plt.savefig("Graph.pdf")
plt.show()
