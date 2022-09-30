import sys
import math
from collections import Counter
import string
import numpy as np

mylist = []
def find_F_lang(mylist,e,prob):
    myar = np.array(mylist)
    er = np.array(e)
    er = np.log(er)
    a = np.inner(myar,er) + np.log(prob)
    print('%0.4f' %a)
    return a

def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X=dict()
    with open (filename,encoding='utf-8') as f:
        # TODO: add your code here
        st = f.read()
        st = st.upper()
        #print(st)
        for char in string.ascii_uppercase:
            X[char] = st.count(char)
            print(char,X[char])
            mylist.append(st.count(char))

    return X



# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!
print("Q1")
X = shred("letter.txt")

print("Q2")
e,s = get_parameter_vectors()
char = 'A'
print('%0.4f' % (X[char] * np.log(e[ord(char)-ord('A')])))
print('%0.4f' % (X[char] * np.log(s[ord(char)-ord('A')])))

print("Q3")
f_eng = find_F_lang(mylist,e,prob= 0.6)
f_spa = find_F_lang(mylist,s,prob= 0.4)

if f_spa - f_eng >= 100:
    p_eng = 0
elif f_spa - f_eng <= -100:
    p_eng = 1
else:
    p_eng = 1/(1 + np.exp(f_spa-f_eng))
    
print("Q4")
print('%0.4f' %p_eng)
