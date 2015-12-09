# coding=utf-8
from PIL import Image, ImageSequence
import sys, os

from PIL.ImageDraw import ImageDraw
from PIL.ImageFont import truetype

outputdir = "test/Arch_sigmaS/"

#Read images and IQA
images = {float(x[0:len(x)-4]): os.path.join(outputdir, x) for x in os.listdir(outputdir) if x.lower().endswith("png")}
#iqas = {x: float(open(y+"_iqa.txt", 'r').read()) for x,y in images.iteritems()}
images = {x: Image.open(y) for x,y in images.iteritems()}

#Annotate images
def annotate(image, text):
    draw = ImageDraw(image)
    font = truetype("Helvetica.otf", 16)
    draw.text((0, 0), text, (255, 255, 255), font=font)

idx = 0
for x in sorted(images.keys()):
    annotate(images[x], "sigmaS="+str(x))#+";IQA="+str(iqas[x]))
    images[x].save(os.path.join(outputdir, str(idx).zfill(4)+"_annotated.png"))
    idx += 1

os.system("convert -delay 100 "+str(outputdir)+"*_annotated.png "+str(outputdir)+"animated.gif")
os.system("rm " + str(outputdir) + "*_annotated.png")