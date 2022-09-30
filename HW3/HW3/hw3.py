import scipy
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from numpy import linalg as LA
from matplotlib import transforms

def load_and_center_dataset(filename):
    # Your implementation goes here!
    x = np.load(filename)
    print(x.shape)
    mn = np.mean(x, axis = 0)
    print(mn.shape)
    x = x - mn
    print(np.average(x))
    return x
    pass

def get_covariance(dataset):
    # Your implementation goes here!
    ds = np.transpose(dataset)
    a = (np.dot(ds,dataset))/(dataset.shape[0]-1)
    '''a = np.dot(dataset, np.transpose(dataset))
    a = np.dot(np.transpose(a), a)
    n = a.shape[0]
    a = a/(n-1)
    print(a.shape)'''
    return a
    pass

def get_eig(S, m):
    a,b = scipy.linalg.eigh(S,subset_by_index=[1024-m, 1024-1])
    return a,b
    # Your implementation goes here!
    pass

def get_eig_prop(S, prop):
    # Your implementation goes here!
    pass

def project_image(image, U):
    # Your implementation goes here!
    '''Ut = np.transpose(U)
    alpha = Ut * image 
    print(alpha.shape)
    ans0 = alpha[0] *  (Ut[0])
    ans1 = alpha[1] * (Ut[1])
    ans = ans0 + ans1'''
    #ans = np.dot(alpha.T,Ut)
    

    ans = np.dot(np.transpose(U),image)
    print("ans shape ",ans)
    temp= np.matmul(U,ans)
    

    print(temp.shape)
    return temp

    pass

def display_image(orig, proj):
    # Your implementation goes here!
    '''fig, axes = plt.subplots(nrows=1, ncols=2)

    #minmin = np.min([np.min(orig), np.min(proj)])
    #maxmax = np.max([np.max(orig), np.max(proj)])

    

    im1 = axes[0].imshow(orig.reshape(32,32) ,aspect='equal', label ="original",extent=(0,30,30,0))
    fig.colorbar(im1, orientation='vertical')
    #im1.show()

    im2 = axes[1].imshow(proj.reshape(32,32) ,aspect='equal', label = "projection",extent=(0,30,30,0))
    fig.colorbar(im2, orientation='vertical')

    plt.show()'''
    
    plt.subplot(1, 2, 1)
    plt.imshow(orig.reshape(32,32) ,aspect='equal', label ="original",extent=(0,30,30,0))
    plt.colorbar(orientation='vertical')
    plt.subplot(1, 2, 2)
    plt.imshow(proj.reshape(32,32) ,aspect='equal', label = "projection",extent=(0,30,30,0))
    plt.colorbar(orientation='vertical')
    plt.show()
    pass

x = load_and_center_dataset("YaleB_32x32.npy")
S = get_covariance(x)
print(S.shape)

'''print("break")
SS = np.cov(x.T)
print(SS)'''

#print(S.shape)
lamda, U = get_eig(S,2)
#print(lamda.shape)
#print(U.shape)
#print(U)


lamda = lamda[::-1]
#U = U[::-1]
lamda = np.diag(lamda)
U = np.fliplr(U)
print(U.shape)

#X = load_and_center_dataset("YaleB_32x32.npy")
projection = project_image(x[0], U)
print(projection)
display_image(x[0], projection)