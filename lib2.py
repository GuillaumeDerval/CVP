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
    # Formula (5) + mean
    w=numpy.zeros((len(imgs), len(imgs[0]) , len(imgs[0][0])))
    i=0
    for img in imgs:
        l, a, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        w[i] = (1-(2*l/255.0-1)**12+1-(2*a/255.0-1)**12+1-(2*b/255.0-1)**12)/3
        i=i+1
    return w    

def kernel_density_est(x):
    # formula (4)
    H = numpy.identity(5)
    return numpy.linalg.det(H)**(-0.5)*2*numpy.pi**(5.0/2)*numpy.exp(-0.5*numpy.dot(numpy.dot(numpy.transpose(x), numpy.linalg.inv(H)), x))
    
def background_prob(imgs, w, p, q):
    # formula (6)
    
    prob = numpy.zeros((len(w), len(w[0]), len(w[0][0])))
    
    for r in range(0, len(imgs)):
        print r
        for y in range(0, len(imgs[0])):
            print y
            for x in range(0, len(imgs[0][0])):
                #print x
                pix = numpy.transpose(numpy.array([x, y, imgs[r][y][x][0], imgs[r][y][x][1], imgs[r][y][x][2]]))
                
                sum1 = 0.0 #nominator
                sum2 = 0.0 #denominator
                
                for s in range(0, len(imgs)):
                    for ip in range(max(0, y-q), min(y+q, len(imgs[0]))):
                        for iq in range(max(0, x-p), min(x+p, len(imgs[0][0]))):
                            if((x,y)!=(ip,iq)):
                                sum1 += kernel_density_est(pix - w[s][ip][iq]*numpy.transpose(numpy.array([iq, ip, imgs[s][ip][iq][0], imgs[s][ip][iq][1], imgs[s][ip][iq][2]])))
                                sum2 += w[s][ip][iq]

                prob[r][y][x] = sum1/sum2
                
    return prob
    
def compute_hdr(imgs,weight_method=hat_weights):
    # Convert all images to Lab color space
    imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2Lab) for img in imgs]
    
    # initial computation
    w_init = weight_method(imgs)
    W = w_init
    
    # iterations (formula (7))
    for i in range(0,3):
        W = w_init*background_prob(imgs, W, 2, 2)
    
    # Apply same weight to all color coordinates
    W = normalise(W)
    Ws = map(lambda x: numpy.dstack((x, x, x,)), W)
    tmp = map(lambda x: (Ws[x]*imgs[x]), range(0, len(imgs)))
    img =  numpy.uint8(sum(tmp)) 
    # Returns image converted to RGB space
    return cv2.cvtColor(img, cv2.COLOR_Lab2RGB)