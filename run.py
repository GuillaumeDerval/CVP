# coding=utf-8
import numpy

from scipy import ndimage
from scipy import misc
import os
import matplotlib.pyplot as plt
from lib import compute_static_hdr, ntsc_grayscale, srgb_grayscale, stupid_grayscale, compute_m_theta, compute_dynamic_hdr, compute_dynamic_lab_hdr

static_img_path = {
    "chateau":      ["img/chateau/chateau0.jpg",
                     "img/chateau/chateau2.jpg",
                     "img/chateau/chateau4.jpg",
                     "img/chateau/chateau-2.jpg",
                     "img/chateau/chateau-4.jpg"],
    "memorial":     ["img/Memorial/memorial0061.png",
                     "img/Memorial/memorial0062.png",
                     "img/Memorial/memorial0063.png",
                     "img/Memorial/memorial0064.png",
                     "img/Memorial/memorial0065.png",
                     "img/Memorial/memorial0066.png",
                     "img/Memorial/memorial0067.png",
                     "img/Memorial/memorial0068.png"],
    "grandcanal":   ["img/grandcanal/grandcanal_mean.jpg",
                     "img/grandcanal/grandcanal_over.jpg",
                     "img/grandcanal/grandcanal_under.jpg"],
    "mask":         ["img/Mask/mask_mean.jpg",
                     "img/Mask/mask_over.jpg",
                     "img/Mask/mask_under.jpg"]
}

dynamic_img_path = {
    "arch": ["img/ArchSequence/A_1.jpg",
             "img/ArchSequence/A_2_03.jpg",
             "img/ArchSequence/A_3_02.jpg",
             "img/ArchSequence/A_4_06.jpg",
             "img/ArchSequence/A_5_01.jpg"],
    "forrest": ["img/ForrestSequence/1_Scene7_10.tif",
                "img/ForrestSequence/2_Scene7_09.tif",
                "img/ForrestSequence/3_Scene7_05.tif",
                "img/ForrestSequence/4_Scene7_13.tif"],
    "puppets": ["img/PuppetsSequence/npGhost_1.tif",
                "img/PuppetsSequence/npGhost_2.tif",
                "img/PuppetsSequence/npGhost_3.tif",
                "img/PuppetsSequence/npGhost_4.tif",
                "img/PuppetsSequence/npGhost_5.tif"],
    "garden": ["img/SculptureGardenSequence/1.tif",
               "img/SculptureGardenSequence/2.tif",
               "img/SculptureGardenSequence/3.tif",
               "img/SculptureGardenSequence/4.tif",
               "img/SculptureGardenSequence/5.tif"]
}

def output_gradients(name, sigma):
    imgs = map(misc.imread, static_img_path[name])
    imgs_bw = map(srgb_grayscale, imgs)
    #Ixs, Iys, ms, thetas = zip(*map(lambda x: compute_m_theta(x, sigma), imgs_bw))

    #if not os.path.exists("output/{}".format(name)):
    #    os.mkdir("output/{}".format(name))

    #for i in range(0, len(imgs)):
    #    misc.imsave('output/{}/{}_{}_sigma{}.png'.format(name, str(i), "orig", sigma), numpy.uint8(imgs[i]))
    #    misc.imsave('output/{}/{}_{}_sigma{}.png'.format(name, str(i), "dx", sigma), numpy.uint8(Ixs[i]))
    #    misc.imsave('output/{}/{}_{}_sigma{}.png'.format(name, str(i), "dy", sigma), numpy.uint8(Iys[i]))
    #    misc.imsave('output/{}/{}_{}_sigma{}.png'.format(name, str(i), "m", sigma), numpy.uint8(ms[i]*255.0))
    #    misc.imsave('output/{}/{}_{}_sigma{}.png'.format(name, str(i), "theta", sigma), numpy.uint8(255.0*(thetas[i]+numpy.pi)/(2.0*numpy.pi)))
    output = compute_static_hdr(imgs, sigma)
    displayable_output = numpy.uint8(output)
    misc.imsave('output/{}_{}.png'.format(name, sigma), displayable_output)


#output_gradients("grandcanal", 2)

#for name in static_img_path:
#    for i in [2, 3, 5, 10, 20, 50]:
#        output_gradients(name, i)

def output_img(name, sigma=2):
    imgs = map(misc.imread, dynamic_img_path[name])
    output = compute_dynamic_hdr(imgs, sigma)
    displayable_output = numpy.uint8(output)
    misc.imsave('output/{}_sigma{}g.png'.format(name, sigma), displayable_output)

    # plt.imshow(displayable_output)
    # plt.show()

def output2_img(name):
    import lib2
    imgs = map(misc.imread, dynamic_img_path[name])
    output = lib2.compute_hdr(imgs)
    displayable_output = numpy.uint8(output)
    misc.imsave('output/{}_hat_backprob.png'.format(name), displayable_output)
	
def output3_img(name, sigma=2):
    imgs = map(misc.imread, dynamic_img_path[name])
    output = compute_dynamic_lab_hdr(imgs, sigma)
    displayable_output = numpy.uint8(output)
    misc.imsave('output/{}_lab_sigma{}g.png'.format(name, sigma), displayable_output)
    
#output_img('puppets')
output3_img('puppets')
#output_img('forrest',2)
#output2_img('puppets')
