# coding=utf-8
import copy
import numpy
import cv2
from scipy import ndimage
import scipy
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, generic_gradient_magnitude, convolve
from scipy import signal

def normalise(ms, epsilon=1e-15):
    """
    Compute V_i for all images from the arrays m_i
    :param ms: array of the arrays m_i
    :return: an array containing all the arrays V_i
    """
    deno = sum(ms) + epsilon

    return [m/deno for m in ms]

def hat_weights(imgs):
    w=numpy.zeros((len(imgs), len(imgs[0]) , len(imgs[0][0])))
    i=0
    for img in imgs:
        l, a, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        w[i] = (1-(2*l/255.0-1)**12+1-(2*a/255.0-1)**12+1-(2*b/255.0-1)**12)/3
        i=i+1
    return w    

def background_prob(imgs, w, p, q):
    # dummy probabilities
    return numpy.ones((len(w), len(w[0]), len(w[0][0])))
    
def compute_hdr(imgs,weight_method=hat_weights):
    # Convert all images to Lab color space
    imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2Lab) for img in imgs]
    
    # initial computation
    w_init = weight_method(imgs)
    W = w_init
    
    # iterations
    for i in range(0,5):
        W = w_init*background_prob(imgs, W, 3, 3)
    
    # Apply same weight to all color coordinates
    W = normalise(W)
    Ws = map(lambda x: numpy.dstack((x, x, x,)), W)
    tmp = map(lambda x: (Ws[x]*imgs[x]), range(0, len(imgs)))
    img =  numpy.uint8(sum(tmp)) 
    # Returns image converted to RGB space
    return cv2.cvtColor(img, cv2.COLOR_Lab2RGB)