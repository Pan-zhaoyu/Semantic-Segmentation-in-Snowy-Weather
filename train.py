from keras.applications.vgg16 import preprocess_input
from PIL import Image
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skimage.io as io
from Basic import generate_arrays_from_file, writeImage
from FCN import fcn_32s
import Segnet_model
import Segnet_test_model
import Unet_model

nb_classes = 20

#path = '/home/robotics/PycharmProjects/Database/train.txt'
#val_path = '/home/robotics/PycharmProjects/Database/val.txt'
#
#input_path = '/home/robotics/PycharmProjects/Database/Train/'
#val_in_path = '/home/robotics/PycharmProjects/Database/Val/'
#output_path = '/home/robotics/PycharmProjects/Database/Train-label/'
#val_out_path = '/home/robotics/PycharmProjects/Database/Val-label/'

path = '/home/robotics/PycharmProjects/Mydata/train.txt'
val_path = '/home/robotics/PycharmProjects/Mydata/val.txt'
input_path = '/home/robotics/PycharmProjects/Mydata/Train/'
val_in_path = '/home/robotics/PycharmProjects/Mydata/Val/'
output_path = '/home/robotics/PycharmProjects/Mydata/train-label/'
val_out_path = '/home/robotics/PycharmProjects/Mydata/val-label/'

model = fcn_32s()
#model = Segnet_model.SegNet()
#model = Segnet_test_model.SegNet()
#model = Unet_model.Unet()

model.compile(loss="binary_crossentropy", optimizer='sgd', metrics=['accuracy'])
BS = 1
EPOCHS = 1
H = model.fit_generator(
        generator=generate_arrays_from_file(path,input_path,output_path),
        steps_per_epoch=524//BS,
        epochs=EPOCHS,
    verbose=1,
    validation_data=generate_arrays_from_file(val_path, val_in_path, val_out_path),
    validation_steps=175//BS)

model.save('/home/robotics/PycharmProjects/RESULTS/fcn_model_0603.h5')
#yaml_string = model.to_yaml()
#model = model_from_yaml(yaml_string)
#print(yaml_string)

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label= "train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label= "train_acc", color = 'blue')
plt.plot(np.arange(0, N), H.history["val_acc"], label = "val_acc", color = 'orangered')
plt.title("Training Loss and Accuracy on FCN")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc = "lower left")
plt.savefig("/home/robotics/PycharmProjects/plot_0603")

#data_dir = '/home/robotics/PycharmProjects/Database/Test'
data_dir = '/home/robotics/PycharmProjects/Mydata/Test'
str = data_dir + '/*.jpg'
test = io.ImageCollection(str)
pred_dir = '/home/robotics/PycharmProjects/predict_results/FCN_0603/'

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
    io.imsave(pred_dir + '201806' + np.str(i) + '.png', image_test)

duration = time.time() - start_time
print('{}s used to predict.\n'.format(duration))