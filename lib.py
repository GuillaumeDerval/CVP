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

def compute_m_theta(img, sigma=2):
    """
    Compute Ix, Iy, m and theta
    :param img:
    :return: m and theta
    """

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
    id = numpy.zeros((2 * l + 1, 2*l+1)) + 1

    for i in range(0, len(thetas)):
        for j in range(i+1, len(thetas)):
            tt = abs(thetas[i] - thetas[j])
            t = scipy.ndimage.filters.convolve(tt, id)
            d[(i, j)] = t/float((2*l+1)**2)
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
    C = numpy.copy(S)
    tmin = (1.0 - tolerance) * 255.0
    tmax = 255.0*tolerance

    for i, img in enumerate(imgs_wb):
        img = (img > tmin) & (img < tmax)
        C[i] *= img

    C = normalise(C)

    return C


def cross_bilinear(img_wb, W, sigmaColor=255.0/10, sigmaSpace=16.0):
    joint = numpy.float32(img_wb)
    src = numpy.float32(W)
    out = cv2.ximgproc.jointBilateralFilter(joint, src, d=-1, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    return out


def compute_weights_static(imgs_wb, sigma=2):
    _, _, ms, _ = zip(*map(lambda x: compute_m_theta(x, sigma), imgs_wb)) #calcul de l'intensité du gradient

    return ms


def compute_weights_dynamic(imgs_wb, sigma=2, l=9, sigma_s=0.2, sigma_T=2):
    _, _, ms, _ = zip(*map(lambda x: compute_m_theta(x, sigma), imgs_wb))  # calcul de l'intensité du gradient
    _, _, _, thetas = zip(*map(lambda x: compute_m_theta(x, sigma_T), imgs_wb))  # calcul de la direction du gradient

    # Calcul de C, indiquant la variation de gradient de l'image par rapport à d'autres
    d = compute_theta_diff(thetas, l)
    S = compute_S(len(imgs_wb), d, sigma_s)
    C = compute_C(imgs_wb, S, 0.9)

    return map(lambda x: ms[x]*C[x], range(0, len(imgs_wb)))

def filter_weights(imgs_wb, weights, sigmaColor=255.0/10, sigmaSpace=16.0):
    return [cross_bilinear(imgs_wb[i], weights[i], sigmaColor, sigmaSpace) for i in range(0, len(weights))]

def get_image_with_exposure_correction(imgs, filtered_and_normalized_weigths):
    weights = map(lambda x: numpy.dstack((x, x, x,)), filtered_and_normalized_weigths)
    return map(lambda x: (weights[x] * imgs[x]), range(0, len(imgs)))


def extract(img):
    l, a, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return (l, a, b)

def pipeline(imgs,
             sigma,
             sigmaColor,
             sigmaSpace,
             static=True, filter_by_color=False, debug=False, opts=None):

    if not opts:
        opts = {}
    imgs_wb = [srgb_grayscale(img) for img in imgs]
    if static:
        weights = compute_weights_static(imgs_wb, sigma=sigma)
    else:
        weights = compute_weights_dynamic(imgs_wb, sigma=sigma, **opts)

    weights = normalise(weights)

    if filter_by_color:
        weights = [numpy.dstack((x, x, x,)) for x in weights]
        weights = filter_weights(imgs, weights, sigmaColor, sigmaSpace)
        weights = normalise(weights)
    else:
        weights = filter_weights(imgs_wb, weights, sigmaColor, sigmaSpace)
        weights = normalise(weights)
        weights = [numpy.dstack((x, x, x,)) for x in weights]

    img = sum([weights[i] * imgs[i] for i in range(0, len(imgs))])

    return img

# def dynamic_weight_map(imgs,epsilon = 1e-25,sigma=2, l=9, sigma_s=0.2, show_plots=False):
#     _, _, ms, thetas = zip(*map(lambda x: compute_m_theta(x, sigma), imgs))
#     d = compute_theta_diff(thetas, l)
#     S = compute_S(len(imgs), d, sigma_s)
#     C = compute_C(imgs, S, 0.9)
#     Vs = normalise(ms)
#     W = [c*v for c, v in zip(C, Vs)]
#     W = normalise(W)
#     W = [cross_bilinear(imgs[i], W[i]) for i in range(0, len(W))]
#     W = normalise(W)
#     #SW = sum(W) + epsilon
#     #W = [w/SW for w in W]
#
#     if show_plots:
#         fig = plt.figure()
#         for i in range(0, 5):
#             a = fig.add_subplot(4, 5, i+1)
#             plt.imshow(imgs[i]/255., cmap=plt.cm.gray)
#             a = fig.add_subplot(4, 5, i+5+1)
#             plt.imshow((thetas[i]+numpy.pi)/(2.0* numpy.pi), cmap=plt.cm.gray)
#             a = fig.add_subplot(4, 5, i + 10 + 1)
#             plt.imshow(C[i], cmap=plt.cm.gray)
#             a = fig.add_subplot(4, 5, i+15+1)
#             plt.imshow(W[i], cmap=plt.cm.gray)
#             fig.add_subplot(1, 5, i+1)
#             cbar = fig.colorbar(plt.imshow(thetas[i]))
#
#         plt.show()
#
#     return W
#
# def compute_dynamic_lab_hdr(imgs, sigma=2, l=9, sigma_s=0.2):
#
#     for i in range(0, len(imgs)):
#         imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2LAB)
#
#     return cv2.cvtColor(pipeline_basic(imgs, True), cv2.COLOR_LAB2RGB)
