from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.initializers import Constant
from keras.applications.vgg16 import VGG16
import matplotlib
matplotlib.use("Agg")
from Basic import bilinear_upsample_weights

nb_classes = 20

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
    model.compile(loss="binary_crossentropy", optimizer='sgd', metrics=['accuracy'])  
    return model
