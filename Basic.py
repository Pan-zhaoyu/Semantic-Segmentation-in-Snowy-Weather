from keras.applications.vgg16 import preprocess_input
from PIL import Image
import numpy as np
import os
import matplotlib
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
    #img = img_org.resize(((320//32)*32, (224//32)*32))
    img = img_org.resize(((w//32)*32, (h//32)*32))
    img = np.array(img, dtype=np.float32)
    x = np.expand_dims(img, axis=0) #np.expand_dims(a, axis)
    x = preprocess_input(x)
    return x

def load_label(path):
    img_org = Image.open(path)
    w, h = img_org.size
    #img = img_org.resize(((320//32)*32, (224//32)*32))
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
    Car = [0, 0, 142]  # 5
    Sky = [70, 130, 180]  # 6
    Roadway = [128, 63, 127]  # 7
    Sidewalk = [243, 35, 232]  # 8
    SnowMass = [81, 0, 81]  # 9
    Vegetation = [106, 142, 34]  # 10
    Person = [217, 22, 56]  # 11
    Animal = [0, 128, 0]    # 2
    Building = [70, 70, 70]  # 13
    TrafficSign = [220, 220, 0]  # 14
    TrafficLight = [192, 192, 128]  # 12
    TelegraphPole = [157, 149, 160]  # 16
    Truck = [64, 128, 255]  # 15
    Bus = [128, 64, 255]    # 1
    Field = [64, 128, 128]  # 3
    SnowBlowing = [189, 153, 153]  # 4
    Manhole = [63, 64, 64]  # 17
    Unlabelled = [0, 0, 0]  # 18
    unlabelled1 = [0, 0, 0]
    unlabelled2 = [0, 0, 0]

    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array([Bus, Animal, Field, SnowBlowing, Car,
                              Sky, Roadway, Sidewalk, SnowMass, Vegetation,
                              Person, TrafficLight, Building, TrafficSign, Truck,
                              TelegraphPole, Manhole, Unlabelled, unlabelled1, unlabelled2])
    for l in range(0, nb_classes):
        r[image == l] = label_colours[l, 0]
        g[image == l] = label_colours[l, 1]
        b[image == l] = label_colours[l, 2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:, :, 0] = r / 1.0
    rgb[:, :, 1] = g / 1.0
    rgb[:, :, 2] = b / 1.0
    im = Image.fromarray(np.uint8(rgb))
    return im