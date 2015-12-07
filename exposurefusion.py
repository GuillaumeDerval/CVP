#! /usr/bin/python
# coding=utf-8
import os
from scipy import misc

from lib import *
import argparse

def pipeline(imgs, static=True, filter_by_color=False, debug=False, opts=None):
    if not opts:
        opts = {}
    imgs_wb = [srgb_grayscale(img) for img in imgs]
    if static:
        weights = compute_weights_static(imgs_wb, **opts)
    else:
        weights = compute_weights_dynamic(imgs_wb, **opts)

    weights = normalise(weights)

    if filter_by_color:
        weights = [numpy.dstack((x, x, x,)) for x in weights]
        weights = filter_weights(imgs, weights)
        weights = normalise(weights)
    else:
        weights = filter_weights(imgs_wb, weights)
        weights = normalise(weights)
        weights = [numpy.dstack((x, x, x,)) for x in weights]

    img = sum([weights[i] * imgs[i] for i in range(0, len(imgs))])

    return img

parser = argparse.ArgumentParser(description='Compute an exposure-fused image from multiple different exposures')
parser.add_argument("source", help="Folder containing all the images to be fused together")
parser.add_argument("output", help="Where to put the output")
parser.add_argument("--debug", help="Display intermediate results", action="store_true")
parser.add_argument("--dynamic", help="Run dynamic exposure fusion", action="store_true")
parser.add_argument("--filterbycolor", help="Run the cross bilateral filter on color rather than luminance", action="store_true")

args = parser.parse_args()
images = [misc.imread(os.path.join(args.source,x)) for x in os.listdir(args.source) if
          x.lower().endswith("jpg") or
          x.lower().endswith("png") or
          x.lower().endswith("tif")]
output = pipeline(images, static=(args.dynamic is False), filter_by_color=(args.filterbycolor is True))
displayable_output = numpy.uint8(output)
misc.imsave(args.output, displayable_output)