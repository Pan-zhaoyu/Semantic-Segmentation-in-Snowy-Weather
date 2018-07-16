from pylab import *
from keras.preprocessing.image import *
from PIL import Image
import time
import os

nb_classes = 20 + 1  #because of the tuple

def calculate_iou(nb_classes, res_dir, label_dir, image_list, label_list):
    conf_m = zeros((nb_classes, nb_classes), dtype=float)

    for img_num1, img_num2 in zip(image_list, label_list):
        img_num1 = img_num1.strip('\n')
        img_num2 = img_num2.strip('\n')
        pred = img_to_array(Image.open('%s/%s.png' % (res_dir, img_num1))).astype(int)
        label = img_to_array(Image.open('%s/%s.png' % (label_dir, img_num2))).astype(int)
        flat_pred = np.ravel(pred)
        flat_label = np.ravel(label)
        #print flat_pred[887600:888600]
        for p, l in zip(flat_pred, flat_label):
            if l == 255:
              continue
            if (l < nb_classes) and (p < nb_classes):
                conf_m[l, p] += 1
            else:
                print('Invalid entry encountered, skipping! Label: ', l, ' Img_num: ', img_num1,
                      ' Prediction: ', p, ' Img_num: ', img_num2)


    I = np.diag(conf_m)
    U = np.sum(conf_m, axis=0) + np.sum(conf_m, axis=1) - I
    IOU = I/U
    #pixel = np.sum(np.diag(conf_m))/np.sum(conf_m)
    meanIOU = np.mean(IOU)

    return conf_m, IOU, meanIOU

label_list = open('/home/robotics/PycharmProjects/Mydata/test.txt').readlines()
label_dir = os.path.expanduser('/home/robotics/PycharmProjects/Mydata/test-label')
image_list = open('/home/robotics/PycharmProjects/predict_results/FCN_0704_1/pre.txt').readlines()
res_dir = os.path.expanduser('/home/robotics/PycharmProjects/predict_results/FCN_0704_1')

start_time = time.time()
conf_m, IOU, meanIOU = calculate_iou(nb_classes, res_dir, label_dir, image_list, label_list)
print("IoU: ")
print(IOU)
print("meanIoU: %f" % meanIOU)
#print('pixel acc: %f' % (np.sum(np.diag(conf_m))/np.sum(conf_m)))
#print(pixel)
duration = time.time() - start_time
print('{}s used to calculate IOU.\n'.format(duration))