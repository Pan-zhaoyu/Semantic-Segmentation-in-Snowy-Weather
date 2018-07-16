from keras.applications.vgg16 import preprocess_input
from keras.models import load_model, model_from_yaml
from Basic import writeImage
from PIL import Image
import time
import numpy as np
import skimage.io as io

data_dir = '/home/robotics/PycharmProjects/Mydata/Test/'
str = data_dir + '/*.jpg'
test = io.ImageCollection(str)
pred_dir = '/home/robotics/PycharmProjects/predict_results/FCN_0716/'
#pred_dir1 = '/home/robotics/PycharmProjects/predict_results/FCN_0704_1/'

yaml_file = open("/home/robotics/PycharmProjects/RESULTS/fcn.yaml", "r")
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)
model.load_weights("/home/robotics/PycharmProjects/RESULTS/fcn.h5")
print("Loaded model from disk")

def model_predict(model, img_org):

    w, h = img_org.size
    img = img_org.resize(((w//32)*32, (h//32)*32))
    img = np.array(img, dtype=np.float32)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    pred = model.predict(x)
    pred = pred[0].argmax(axis=-1).astype(np.uint8)
    out = writeImage(pred)
    return out,pred

start_time = time.time()

for i in range(len(test)):
    #print ("[INFO] loading network...")
    image_test, image_pred= model_predict(model, Image.fromarray(test[i]))
    io.imsave(pred_dir + '20180716_' + np.str(i) + '.png', image_test)
    #io.imsave(pred_dir1 + '20180614_' + np.str(i) + '.png', image_pred)

duration = time.time() - start_time

print('{}s used to predict.\n'.format(duration))