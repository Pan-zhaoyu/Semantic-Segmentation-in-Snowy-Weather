from keras.models import Model, model_from_yaml, load_model
from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.initializers import Constant
from keras.applications.vgg16 import VGG16, preprocess_input
from PIL import Image
import time
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skimage.io as io
import Segnet_model
import Segnet_test_model
import Unet_model

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

def fcn_32s():
    inputs = Input(shape=(None, None, 3))
    vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
    x = Conv2D(filters=nb_classes,
               kernel_size=(1, 1))(vgg16.output)
    x = Conv2DTranspose(filters=nb_classes,
                        kernel_size=(64, 64),
                        strides=(32, 32),
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer=Constant(bilinear_upsample_weights(32, nb_classes)))(x)
    model = Model(inputs=inputs, outputs=x)
    for layer in model.layers[:15]:
        layer.trainable = False
#    model.summary()
    return model

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
    unlabelled1 = [0, 0, 1]
    unlabelled2 = [0, 0, 2]
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
path = '/home/robotics/PycharmProjects/Database/train.txt'
val_path = '/home/robotics/PycharmProjects/Database/val.txt'

input_path = '/home/robotics/PycharmProjects/Database/Train/'
val_in_path = '/home/robotics/PycharmProjects/Database/Val/'
output_path = '/home/robotics/PycharmProjects/Database/Train-label/'
val_out_path = '/home/robotics/PycharmProjects/Database/Val-label/'

#path = '/home/robotics/workplace/dataset/train.txt'
#val_path = '/home/robotics/workplace/dataset/val.txt'
#input_path = '/home/robotics/workplace/dataset/train/'
#val_in_path = '/home/robotics/workplace/dataset/val/'
#output_path = '/home/robotics/workplace/dataset/train-labeled/'
#val_out_path = '/home/robotics/workplace/dataset/val-labeled/'

model = fcn_32s()
#model = Segnet_model.SegNet()
#model = Segnet_test_model.SegNet()
#model = Unet_model.Unet()

model.compile(loss="binary_crossentropy", optimizer='sgd', metrics=['accuracy'])
BS = 2
EPOCHS = 300
H = model.fit_generator(
        generator=generate_arrays_from_file(path,input_path,output_path),
        steps_per_epoch=293//BS,
        epochs=EPOCHS,
    verbose=1,
    validation_data=generate_arrays_from_file(val_path, val_in_path, val_out_path),
    validation_steps=113//BS)

model.save('fcn_model.h5')
#yaml_string = model.to_yaml()
#model = model_from_yaml(yaml_string)
#print(yaml_string)

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label= "train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label= "train_acc", color = 'blue')
plt.plot(np.arange(0, N), H.history["val_acc"], label = "val_acc")
plt.title("Training Loss and Accuracy on FCN")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc = "lower left")
plt.savefig("plot1")

data_dir = '/home/robotics/PycharmProjects/Database/Test'
str = data_dir + '/*.jpg'
test = io.ImageCollection(str)
pred_dir = '/home/robotics/PycharmProjects/predict_results/'

#model = model_from_yaml(yaml_string)
#model = load_model('fcn_model.h5')

def model_predict(model, img_org):

    w, h = img_org.size
    img = img_org.resize(((w//32)*32, (h//32)*32))
    img = np.array(img, dtype=np.float32)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    pred = model.predict(x)
    pred = pred[0].argmax(axis=-1).astype(np.uint8)
    out = writeImage(pred)
    return out

start_time = time.time()

for i in range(len(test)):
    #print ("[INFO] loading network...")
    image_test = model_predict(model, Image.fromarray(test[i]))
    io.imsave(pred_dir + '201805' + np.str(i) + '.png', image_test)

duration = time.time() - start_time
print('{}s used to predict.\n'.format(duration))