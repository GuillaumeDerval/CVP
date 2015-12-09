# coding=utf-8
import os
import subprocess
from scipy import misc
from lib import *
from multiprocessing import Pool


source = "img/ArchSequence/"
outputdir = "test/current/"
def run(sigma=20.0, sigmaColor=255.0/10.0, sigmaSpace=16.0, static=True, l=9, sigma_T=1.0, sigma_s=0.2):
    images = [misc.imread(os.path.join(source, x)) for x in os.listdir(source) if
              x.lower().endswith("jpg") or
              x.lower().endswith("png") or
              x.lower().endswith("tif")]
    opts = {}
    if static is False:
        opts["l"] = l
        opts["sigma_T"] = sigma_T
        opts["sigma_s"] = sigma_s
    output = pipeline(images, sigma, sigmaColor, sigmaSpace, static=static, filter_by_color=False, opts=opts)
    misc.imsave(os.path.join(outputdir, str(sigma_s)+".png"), numpy.uint8(output))


def process_iqa(image):
    p = subprocess.Popen("matlab -wait -nodesktop -nosplash -r \"addpath('mef_iqa'); "
                         "iqa('" + source + "', '" + image + "', 1, '', '" + image + "_iqa.txt')\"",
                         shell=True)
    p.wait()

for sigma_s in [0.1*i for i in range(1, 30, 1)]:
    run(sigma=25, sigmaSpace=25, sigmaColor=25, static=False, l=9, sigma_T=1.0, sigma_s=sigma_s)

#images = [os.path.join(outputdir, x) for x in os.listdir(outputdir) if x.lower().endswith("png")]
#p = Pool(8)
#p.map(process_iqa, images)