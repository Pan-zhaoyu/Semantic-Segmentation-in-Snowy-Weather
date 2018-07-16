import matplotlib
matplotlib.use("Agg")
from Basic import generate_arrays_from_file, drawfigure
from FCN import fcn_32s
from Segnet_model import SegNet
#import Segnet_test_model
import Unet_model

nb_classes = 20

path = '/home/robotics/PycharmProjects/Mydata/train.txt'
val_path = '/home/robotics/PycharmProjects/Mydata/val.txt'
input_path = '/home/robotics/PycharmProjects/Mydata/Train/'
val_in_path = '/home/robotics/PycharmProjects/Mydata/Val/'
output_path = '/home/robotics/PycharmProjects/Mydata/train-label/'
val_out_path = '/home/robotics/PycharmProjects/Mydata/val-label/'

model = fcn_32s()
#model = SegNet()
#model = Segnet_test_model.SegNet()
#model = Unet_model.Unet()

BS = 5
EPOCHS = 1
H = model.fit_generator(
        generator=generate_arrays_from_file(path,input_path,output_path),
        steps_per_epoch=524//BS,
        epochs=EPOCHS,
    verbose=1,
    validation_data=generate_arrays_from_file(val_path, val_in_path, val_out_path),
    validation_steps=175//BS)

model_yaml = model.to_yaml()
with open("/home/robotics/PycharmProjects/RESULTS/fcn.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
model.save_weights("/home/robotics/PycharmProjects/RESULTS/fcn.h5")
print("Saved model to disk")

drawfigure(EPOCHS, H)