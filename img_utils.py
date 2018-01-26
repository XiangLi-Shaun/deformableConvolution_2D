import scipy
import scipy.misc
import numpy as np

def imsave(image, path):
    # 分类的
    label_colours = [(0,0,0),(255,0,0),(0,255,0)]
    
    # 分割的
    #label_colours = [(0,0,0),(255,255,255)]
    images = np.ones(list(image.shape)+[3])
    for j_, j in enumerate(image):
        for k_, k in enumerate(j):
            if k < 3:
                images[j_, k_] = label_colours[int(k)]
    scipy.misc.imsave(path, images)


