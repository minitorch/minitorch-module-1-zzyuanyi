"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(a,b):
    result=a*b
    return a*b
def id (x):
    return x
def add(a,b):
    result=a+b
    return result
def neg(x):
    return -x
def lt(a,b):
    if(a<b):
        return True
    else:
        return False
def eq(a,b):
    if(a==b):
        return True
    else:
        return False
def max(a,b):
    if(a>b):
        return a
    else:
        return b
def is_close(a,b):
    if(abs(a-b)<1e-2):
        return True
    else:
        return False
def sigmoid(x):
    if(x>=0):
        result=1.0/(1.0+math.exp(-x))
    else:
        exp_x=math.exp(x)
        result=exp_x/(1.0+exp_x)
    return result
def relu(x):
    if(x<=0):
        return 0
    else:
        return x
def log(x):
    return math.log(x)
def exp(x):
    return math.exp(x)
def inv(x):
    return 1.0/x
def log_back(x,grad_in):
    grad_out=grad_in/x
    return grad_out
def inv_back(x,grad_in):
    grad_out=grad_in/(-x**2)
    return grad_out
def relu_back(x,grad_in):
    if(x<=0):
        return 0
    else:
        grad_out=grad_in
        return grad_out

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(fn):
    def process(ls):
        arr=[]
        for item in ls:
            arr.append(fn(item))
        return arr
    return process
def reduce(fn,start):
    def process(ls):
        ans=start
        for item in ls:
            ans=fn(ans,item)
        return ans
    return process
def zipWith(fn):
    def process(ls1,ls2):
        arr=[]
        for i in range(len(ls1)):
            arr.append(fn(ls1[i],ls2[i]))
        return arr
    return process
def addLists(a,b):
    assert len(a)==len(b),"list length should be the same"
    temp=zipWith(add)
    return temp(a,b)
def negList(a):
    temp=map(neg)
    return temp(a)
def prod(iterable):
    temp=reduce(mul,1)
    return temp(iterable)
def sum(iterable):
    temp=reduce(add,0)
    return temp(iterable)