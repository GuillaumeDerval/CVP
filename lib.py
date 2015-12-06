# coding=utf-8
import copy
import numpy
import cv2
from scipy import ndimage
import scipy
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, generic_gradient_magnitude, convolve
from scipy import signal

def ntsc_grayscale(img):
    red, green, blue = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return 0.2989 * red + 0.5870 * green + 0.1140 * blue


def srgb_grayscale(img):
    red, green, blue = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    y = 0.2126 * red + 0.7152 * green + 0.0722 * blue

    return y

def stupid_grayscale(img):
    red, green, blue = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    y = (1.0*red+1.0*green+1.0*blue)/3.0

    return y


def gauss_derivative_kernels(sigma):
    """ returns x and y derivatives of a 2D
        gauss kernel array for convolutions """
    size = sigma*2

    y, x = numpy.mgrid[-size:size+1, -size:size+1]

    coef = 2.0 * numpy.pi * (sigma ** 4)
    sigma = float(2 * (sigma ** 2))

    # x and y derivatives of a 2D gaussian with standard dev half of size
    # (ignore scale factor)
    gx = - x * numpy.exp(-(((x ** 2) + (y ** 2)) / sigma)) / coef
    gy = - y * numpy.exp(-(((x ** 2) + (y ** 2)) / sigma)) / coef

    return gx, gy


def gauss_derivatives(im, sigma):
    """ returns x and y derivatives of an image using gaussian
        derivative filters of size n. The optional argument
        ny allows for a different size in the y direction."""

    gx, gy = gauss_derivative_kernels(sigma)

    imx = scipy.ndimage.filters.convolve(im, gx)
    imy = scipy.ndimage.filters.convolve(im, gy)

    return imx, imy

def compute_m_theta(img, sigma=2):
    """
    Compute Ix, Iy, m and theta
    :param img:
    :return: m and theta
    """
    #i_x, i_y = gauss_derivatives(img/255., sigma)

    #i_x, i_y = numpy.gradient(img, edge_order=2)

    #i_x = ndimage.filters.gaussian_gradient_magnitude(img/255.0, [sigma, 0])
    #i_y = ndimage.filters.gaussian_gradient_magnitude(img/255.0, [0, sigma])

    #i_x = img
    #i_y = img

    i_x = ndimage.gaussian_filter(img, sigma=sigma, order=[1, 0], output=numpy.float64, mode='nearest')
    i_y = ndimage.gaussian_filter(img, sigma=sigma, order=[0, 1], output=numpy.float64, mode='nearest')

    m = numpy.sqrt((i_x ** 2) + (i_y ** 2))
    theta = numpy.arctan2(i_y, i_x)

    return i_x, i_y, m, theta

def normalise(ms, epsilon=1e-15):
    """
    Compute V_i for all images from the arrays m_i
    :param ms: array of the arrays m_i
    :return: an array containing all the arrays V_i
    """
    deno = sum(ms) + epsilon

    return [m/deno for m in ms]

def compute_theta_diff(thetas, l=9):
    d = {}
    id = numpy.zeros((2 * l + 1, 2*l+1)) + 1#
    #id = numpy.identity(2*l+1)

    for i in range(0, len(thetas)):
        for j in range(i+1, len(thetas)):
            tt = abs(thetas[i] - thetas[j])
            t = scipy.ndimage.filters.convolve(tt, id)
            d[(i, j)] = t/float((2*l+1)**2)
            #d[(i, j)] = t / float((2 * l + 1))
    return d

def compute_S(n, d, sigma_s):
    sigma_s = 2.0*sigma_s**2

    S = [numpy.zeros(numpy.shape(d[(0, 1)])) for i in range(0, n)]
    for i in range(0, n):
        for j in range(i+1, n):
            t = numpy.exp(-(d[(i, j)] ** 2) / sigma_s)
            S[i] += t
            S[j] += t

    return S

def compute_C(imgs_wb, S, tolerance):
    epsilon = 1.0e-25

    C = numpy.copy(S)
    tmin = (1.0 - tolerance) * 255.0
    tmax = 255.0*tolerance

    for i, img in enumerate(imgs_wb):
        #for x in numpy.nditer(img, op_flags=['readwrite']):
        #    x[...] = 0. if x > tmax or x < tmin else 1.
        img = (img > tmin) & (img < tmax)
        C[i] *= img

    SC = sum(C)+epsilon
    for c in C:
        c /= SC

    return C

def cross_bilinear(img_wb, W):
    joint = numpy.float32(img_wb)
    src = numpy.float32(W)
    out = cv2.ximgproc.jointBilateralFilter(joint, src, d=-1, sigmaColor=255/10, sigmaSpace=16)
    return out

def compute_static_hdr(imgs, sigma=2, grayscale_method=ntsc_grayscale):
    imgs_bw = map(grayscale_method, imgs) #convertit l'image en noir et blanc

    i_x, i_y, ms, thetas = zip(*map(lambda x: compute_m_theta(x, sigma), imgs_bw))

    #ms = [    cross_bilinear(imgs_bw[i], ms[i]) for i in range(0, len(ms))]

    Vs = normalise(ms)

    #fig = plt.figure()
    #for i in range(0, 3):
    #    a = fig.add_subplot(3, 6, 6*i + 1)
    #    plt.imshow(imgs_bw[i] / 255., cmap=plt.cm.gray)
    #    a = fig.add_subplot(3, 6, 6*i + 2)
    #    plt.imshow(i_x[i], cmap=plt.cm.gray)
    #    a = fig.add_subplot(3, 6, 6*i + 3)
    #    plt.imshow(i_y[i], cmap=plt.cm.gray)
    #    a = fig.add_subplot(3, 6, 6*i + 4)
    #    plt.imshow(ms[i], cmap=plt.cm.gray)
    #    a = fig.add_subplot(3, 6, 6*i + 5)
    #    plt.imshow(Vs[i], cmap=plt.cm.gray)
    #    a = fig.add_subplot(3, 6, 6 * i + 6)
    #    plt.imshow(cross_bilinear(imgs_bw[i], Vs[i]), cmap=plt.cm.gray)
    #plt.show()

    Vs = [cross_bilinear(imgs_bw[i], Vs[i]) for i in range(0, len(Vs))]
    Vs = normalise(Vs)

    Vss = map(lambda x: numpy.dstack((x, x, x,)), Vs)

    tmp = map(lambda x: (Vss[x]*imgs[x]), range(0, len(imgs)))
    return sum(tmp)

def compute_dynamic_hdr(imgs, sigma=2, l=9, sigma_s=0.2):
    epsilon = 1e-25
    imgs_bw = map(srgb_grayscale, imgs)
    _, _, ms, thetas = zip(*map(lambda x: compute_m_theta(x, sigma), imgs_bw))
    d = compute_theta_diff(thetas, l)
    S = compute_S(len(imgs), d, sigma_s)
    C = compute_C(imgs_bw, S, 0.9)
    Vs = normalise(ms)
    W = [c*v for c, v in zip(C, Vs)]
    W = normalise(W)
    W = [cross_bilinear(imgs_bw[i], W[i]) for i in range(0, len(W))]
    W = normalise(W)
    #SW = sum(W) + epsilon
    #W = [w/SW for w in W]
    #fig = plt.figure()
    #for i in range(0, 5):
        #a = fig.add_subplot(4, 5, i+1)
        #plt.imshow(imgs_bw[i]/255., cmap=plt.cm.gray)
        #a = fig.add_subplot(4, 5, i+5+1)
        #plt.imshow((thetas[i]+numpy.pi)/(2.0* numpy.pi), cmap=plt.cm.gray)
        #a = fig.add_subplot(4, 5, i + 10 + 1)
        #plt.imshow(C[i], cmap=plt.cm.gray)
        #a = fig.add_subplot(4, 5, i+15+1)
        #plt.imshow(W[i], cmap=plt.cm.gray)

    #    fig.add_subplot(1, 5, i+1)

    #    cbar = fig.colorbar(plt.imshow(thetas[i]))
    #plt.show()
    Ws = map(lambda x: numpy.dstack((x, x, x,)), W)
    tmp = map(lambda x: (Ws[x] * imgs[x]), range(0, len(imgs)))
    return sum(tmp)
	
def imgToLab(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    return img
    
def imgFromLab(img):
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return img
	
def extract(img):
    l, a, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return (l, a, b)
    
def dynamic_weight_map(imgs,epsilon = 1e-25,sigma=2, l=9, sigma_s=0.2):
    _, _, ms, thetas = zip(*map(lambda x: compute_m_theta(x, sigma), imgs))
    d = compute_theta_diff(thetas, l)
    S = compute_S(len(imgs), d, sigma_s)
    C = compute_C(imgs, S, 0.9)
    Vs = normalise(ms)
    W = [c*v for c, v in zip(C, Vs)]
    W = normalise(W)
    W = [cross_bilinear(imgs[i], W[i]) for i in range(0, len(W))]
    W = normalise(W)
    #SW = sum(W) + epsilon
    #W = [w/SW for w in W]
    #fig = plt.figure()
    #for i in range(0, 5):
        #a = fig.add_subplot(4, 5, i+1)
        #plt.imshow(imgs[i]/255., cmap=plt.cm.gray)
        #a = fig.add_subplot(4, 5, i+5+1)
        #plt.imshow((thetas[i]+numpy.pi)/(2.0* numpy.pi), cmap=plt.cm.gray)
        #a = fig.add_subplot(4, 5, i + 10 + 1)
        #plt.imshow(C[i], cmap=plt.cm.gray)
        #a = fig.add_subplot(4, 5, i+15+1)
        #plt.imshow(W[i], cmap=plt.cm.gray)

    #    fig.add_subplot(1, 5, i+1)

    #    cbar = fig.colorbar(plt.imshow(thetas[i]))
    #plt.show()
    return W
    
def compute_dynamic_lab_hdr(imgs, sigma=2, l=9, sigma_s=0.2):
    imgs_lab = map(imgToLab, imgs)
    imgs_L = map(lambda img:img[:, :, 0], imgs_lab)
    imgs_alpha = map(lambda img:img[:, :, 1], imgs_lab)
    imgs_beta = map(lambda img:img[:, :, 2], imgs_lab)
    
    W_L = dynamic_weight_map(imgs_L)
    W_a = dynamic_weight_map(imgs_alpha)
    W_b = dynamic_weight_map(imgs_beta)
    
    Ws = map(lambda x: numpy.dstack((W_L[x], W_a[x], W_b[x],)), range(0, len(W_L)))
    #Ws = map(lambda x: numpy.dstack((x, x, x,)), W_L)
    tmp = map(lambda x: (Ws[x] * imgs_lab[x]), range(0, len(imgs_lab)))
    sum_lab = sum(tmp)
    sum_lab=numpy.uint8(sum_lab)
    
    rgb = imgFromLab(sum_lab)
    
    #print(rgb)
    return rgb
