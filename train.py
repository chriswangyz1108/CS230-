import keras
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.applications.resnet50 import ResNet50
from keras.layers import Input
from keras.models import Model
from keras.regularizers import *

# To change model just change line 5 and line 13, 14 properly. for example replace 5 with 
# from keras.applications.inception_v3 import InceptionV3 and every instance of ResNet for InceptionV3
def get_model():
    Input_1 = Input(shape=(3, 200, 200), name='Input_1')
    ResNet50_1_model = ResNet50(include_top= False, input_tensor = Input_1)
    ResNet50_1 = ResNet50_1_model(Input_1)
    num_layers = len(ResNet50_1_model.layers)
    for i, layer in enumerate(ResNet50_1_model.layers):
        if ((i * 100) / (num_layers - 1)) <= (100 - 50):
            layer.trainable = False
    Flatten_1 = Flatten(name='Flatten_1')(ResNet50_1)
    Dense_1 = Dense(name='Dense_1',output_dim= 4096,activation= 'relu' )(Flatten_1)
    Dropout_1 = Dropout(name='Dropout_1',p= .5)(Dense_1)
    Dense_2 = Dense(name='Dense_2',output_dim= 2048,activation= 'relu' )(Dropout_1)
    Dropout_2 = Dropout(name='Dropout_2',p= 0.5)(Dense_2)
    Dense_3 = Dense(name='Dense_3',output_dim= 6,activation= 'softmax' )(Dropout_2)

    model = Model([Input_1],[Dense_3])
    return model

from keras.preprocessing.image import ImageDataGenerator
K.set_image_dim_ordering("th")
# dimensions.
img_width, img_height = 200, 200

train_data_dir = './train'
validation_data_dir = './validation'
nb_train_samples = 1947
nb_validation_samples = 486
epochs = 20
batch_size = 32

input_shape = (3, img_width, img_height)

model = get_model()
model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

# Augmentation
train_datagen = ImageDataGenerator(
	rotation_range=0.2,
    shear_range=0.2,
    height_shift_range=0.2,
	width_shift_range=0.2,
    horizontal_flip=True,)

validation_datagen = ImageDataGenerator(horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    nb_epoch=50,
	samples_per_epoch=nb_train_samples,
    validation_data=validation_generator,
	nb_val_samples=nb_validation_samples)

model.save_weights('model.h5')