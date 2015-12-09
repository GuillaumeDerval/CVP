#! /usr/bin/python
# coding=utf-8
import os
import subprocess
from scipy import misc

from lib import *
import argparse

parser = argparse.ArgumentParser(description='Compute an exposure-fused image from multiple different exposures')
parser.add_argument("source", help="Folder containing all the images to be fused together")
parser.add_argument("output", help="Where to put the output")
parser.add_argument("--debug", help="Display intermediate results", action="store_true")
parser.add_argument("--dynamic", help="Run dynamic exposure fusion", action="store_true")
parser.add_argument("--filterbycolor", help="Run the cross bilateral filter on color rather than luminance", action="store_true")
parser.add_argument("--withiqa", help="Run IQA after fusion", action="store_true")
parser.add_argument("--sigma", help="Fix the sigma for the gaussian derivate to compute the gradients", type=float, default=20.0)
parser.add_argument("--sigmaColor", help="Fix the sigma for colors in the cross bilateral filter", type=float, default=255.0/10.0)
parser.add_argument("--sigmaSpace", help="Fix the sigma for space in the cross bilateral filter", type=float, default=25.0)

args = parser.parse_args()
images = [misc.imread(os.path.join(args.source,x)) for x in os.listdir(args.source) if
          x.lower().endswith("jpg") or
          x.lower().endswith("png") or
          x.lower().endswith("tif")]
output = pipeline(images, args.sigma, args.sigmaColor, args.sigmaSpace, static=(args.dynamic is False), filter_by_color=(args.filterbycolor is True))
displayable_output = numpy.uint8(output)
misc.imsave(args.output, displayable_output)

if args.withiqa:
    DEVNULL = open(os.devnull, 'wb')
    p = subprocess.Popen("matlab -wait -nodesktop -nosplash -r \"addpath('mef_iqa'); "
                         "iqa('" + args.source + "', '" + args.output + "', 1, '"+ args.output+"_iqa.fig', '" + args.output + "_iqa.txt')",
                         shell=True, stdout=DEVNULL, stderr=DEVNULL)
    p.wait()