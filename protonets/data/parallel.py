import numpy as np
import functools
import multiprocessing as mp
import psutil
#from .timer import timer



def get_physical_cores():
    return psutil.cpu_count(logical = False)


def get_chunks(alist, ncores=4):
    nplist=np.array(alist)
    n=len(nplist)
    #n=1198
    #ncores=4
    k=int(n/ncores)
    k
    r=n-k*ncores
    r
    input_load=np.full((ncores,),fill_value=k)
    #print(k,r,input_load)
    
    #distribute remainder uniformly
    for i in range(r):
        #print(i)
        input_load[i]+=1
    #print(input_load,input_load.sum())

    indexes={}
    x=0
    for i in range(ncores):

        indexes[i]=np.arange(x,x+input_load[i])
        #print(i,x,x+input_load[i],indexes[i].shape[0])
        x+=input_load[i]
        
    chunks=[]
    for i in range(ncores):
        chunks.append(nplist[indexes[i]])
        
    return chunks

#@timer
def get_chunks_by_load(alist, weights, ncores=4):
    
    weights_sorted =  np.argsort(weights)
    a1=weights_sorted.copy()
    n=len(a1)
    a2=np.flip(a1,0)
    load_pairs=a1+a2
    n2=int(n/2)
    
    aa1=get_chunks(a1[:n2].tolist(),ncores=ncores)
    aa2=get_chunks(a2[:n2].tolist(),ncores=ncores)
    pload=get_chunks(load_pairs[:n2].tolist(),ncores=ncores)
    
    l=[]
    load=[]
    for l1,l2,l3 in zip(aa1,aa2,pload):
        l.append(np.concatenate((l1,l2),axis=0))
        load.append(l3.sum())
        
    if n%2==0:
        middle=[]
    else:
        middle=np.array([a1[n2]])
        l[-1]=np.concatenate((l[-1],middle),axis=0)
        load[-1]+=a1[n2]

    chunks=[]
    for array in l:
        chunks.append(alist[array])
    
    return chunks, load