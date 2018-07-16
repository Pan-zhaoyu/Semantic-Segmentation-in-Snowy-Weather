from keras.applications.vgg16 import preprocess_input
from PIL import Image
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

nb_classes = 20

# Bilinear interpolation
def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = factor*2 - factor%2
    factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:filter_size, :filter_size]
    upsample_kernel = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes),
                       dtype=np.float32)
    for i in xrange(number_of_classes):
        weights[:, :, i, i] = upsample_kernel
    return weights

def load_image(path):
    img_org = Image.open(path)
    w, h = img_org.size
    #img = img_org.resize(((480//32)*32, (352//32)*32))
    img = img_org.resize(((w//32)*32, (h//32)*32))
    img = np.array(img, dtype=np.float32)
    x = np.expand_dims(img, axis=0) #np.expand_dims(a, axis)
    x = preprocess_input(x)
    return x

def load_label(path):
    img_org = Image.open(path)
    w, h = img_org.size
    #img = img_org.resize(((480//32)*32, (352//32)*32))
    img = img_org.resize(((w//32)*32, (h//32)*32))
    img = np.array(img, dtype=np.uint8)
    img[img==255] = 0
    y = np.zeros((1, img.shape[0], img.shape[1], nb_classes), dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            y[0, i, j, img[i][j]] = 1
    return y

def generate_arrays_from_file(path, image_dir, label_dir):
    while 1:
        f = open(path)
        for line in f:
            filename = line.rstrip('\n')
            path_image = os.path.join(image_dir, filename+".jpg")
            path_label = os.path.join(label_dir, filename+".png")
            x = load_image(path_image)
            y = load_label(path_label)
            yield (x, y)

        f.close()

def writeImage(image):
    """ label data to colored image """
    unlabelled = [0,0,0]        #1
    ego_vehicle = [0,0,0]       #2
    static_object = [190,153,153]#3
    car = [0,0,142]             #4
    sky = [70,130,180]          #5
    roadway = [128,64,128]      #6
    sidewalk = [244,35,232]     #7
    snow_mass = [81,0,81]       #8
    vegetation = [107,142,35]   #9
    person = [220,20,60]        #10
    animal = [255,0,0]          #11
    building = [70,70,70]       #12
    traffic_sign = [220,220,0]  #13
    traffic_light = [250,170,30]#14
    telegraph_pole = [153,153,153]#15
    truck = [0,0,70]            #16
    bus = [0,60,100]            #17
    field = [152,251,152]       #18
    snow_blowing = [255,255,255]#19
    manhole = [81,0,81]         #20

    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array([unlabelled, ego_vehicle, static_object,car,sky,roadway,sidewalk,snow_mass,vegetation,
                              person,animal,building,traffic_sign,traffic_light,telegraph_pole,
                              truck,bus,field,snow_blowing,manhole])
    for l in range(1, nb_classes+1):
        r[image == l] = label_colours[l-1, 0]
        g[image == l] = label_colours[l-1, 1]
        b[image == l] = label_colours[l-1, 2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:, :, 0] = r / 1.0
    rgb[:, :, 1] = g / 1.0
    rgb[:, :, 2] = b / 1.0
    im = Image.fromarray(np.uint8(rgb))
    return im

def drawfigure(N, H, savepath):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc", color='blue')
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc", color='red')
    plt.title("Training Loss and Accuracy on FCN")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="center")
    plt.savefig(savepath)
