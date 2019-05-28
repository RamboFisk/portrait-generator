import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras import Sequential
from keras.models import load_model
from keras.layers import Convolution2D, Dense, MaxPooling2D, UpSampling2D, Reshape, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.initializers import RandomNormal


from sklearn.utils import shuffle

lr = 0.0002
b1 = 0.5 # 0.9
color_cannels = 3
init = RandomNormal(mean=0.0, stddev=0.02, seed=42)
bn_mom = 0.1

def create_generative_model():
    model = Sequential()

    model.add(Dense(64, activation='sigmoid', input_shape=(128,)))
    model.add(BatchNormalization())
    
    model.add(Reshape((8,8,1)))
    model.add(Convolution2D(512, (4, 4), strides=(1, 1), activation='relu', padding='same', kernel_initializer=init))
    model.add(BatchNormalization(momentum=bn_mom))
    model.add(UpSampling2D((2,2)))
    model.add(Convolution2D(256, (4, 4), activation='relu', padding='same', kernel_initializer=init))
    model.add(BatchNormalization(momentum=bn_mom))
    model.add(UpSampling2D((2,2)))
    model.add(Convolution2D(128, (4, 4), activation='relu', padding='same', kernel_initializer=init))
    model.add(BatchNormalization(momentum=bn_mom))    
    model.add(UpSampling2D((2,2)))
    model.add(Convolution2D(128, (4, 4), activation='relu', padding='same', kernel_initializer=init))
    model.add(BatchNormalization(momentum=bn_mom))
    
    model.add(Convolution2D(color_cannels, (5, 5), activation='sigmoid', padding='same', kernel_initializer=init))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=b1))
    return model

def create_discriminatory_model():
    model = Sequential()

    model.add(Convolution2D(32, (4, 4), activation='sigmoid', input_shape=(64, 64, color_cannels), padding='same', kernel_initializer=init)) # set params here
    model.add(BatchNormalization(momentum=bn_mom))
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(64, (4, 4), activation='relu', padding='same', kernel_initializer=init))
    model.add(BatchNormalization(momentum=bn_mom))
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(128, (4, 4), activation='relu', padding='same', kernel_initializer=init))
    model.add(BatchNormalization(momentum=bn_mom))
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(256, (4, 4), activation='relu', padding='same', kernel_initializer=init))
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(512, (1, 1), activation='relu', padding='same', kernel_initializer=init))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    set_trainability(model, True)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=b1), metrics=['accuracy'])
    set_trainability(model, False)
    return model

def create_adversial_model(g, d):
    model = Sequential()
    model.add(g)
    model.add(d)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=b1), metrics=['accuracy'])
    print(model.summary())
    return model

def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

load = False
if load:
    dis_model = load_model('D_norm_lfw_20k.h5')
    gen_model = load_model('G_norm_lfw_20k.h5')
else:
    dis_model = create_discriminatory_model()
    gen_model = create_generative_model()
ad_model = create_adversial_model(gen_model, dis_model)

epochs = 20
batch_steps = 1000*epochs
batch_size = 100
gen_bs = batch_size

#image generator
def get_gen(bs):
    seed = 42
    datagen = ImageDataGenerator(data_format="channels_last")
    dataset_gen = datagen.flow_from_directory(
        'lfw_data_smol', #'lfw_data_smol', img_align_celeba
        batch_size=bs, 
        color_mode='rgb',  #alt 'grayscale'
        class_mode=None, 
        target_size=(64,64), 
        seed=seed)
    return dataset_gen



gen = get_gen(batch_size)
for i in range(batch_steps):
    #Generate batch for discriminator
    e = np.array(next(gen))/255
    
    if len(e) < batch_size:
        gen = get_gen(batch_size)
        e = np.array(next(gen))/255
    
    set_trainability(dis_model, True)
    
    batch = e
    label = np.ones(batch_size)
    h_dis = dis_model.train_on_batch(batch, label)
    
    batch = gen_model.predict(np.random.normal(size=(batch_size,128)))
    label = np.zeros(batch_size)
    h_dis = dis_model.train_on_batch(batch, label)
    
    set_trainability(dis_model, False)
    
    gen_batch = np.random.normal(size=(gen_bs,128))
    h_ad = ad_model.train_on_batch(gen_batch, np.ones((gen_bs,1)))
    
    if i % 100 == 0:
        print('Final loss and number of batches',i, h_dis, h_ad)

#save models
gen_model.save('G_norm_lfw_20k.h5')
dis_model.save('D_norm_lfw_20k.h5')

noise = np.random.normal(size=(4, 128))
images = gen_model.predict(noise).reshape((4,64,64,3))

plt.rcParams["figure.figsize"] = (12, 9) # (w, h)

plt.subplot('42'+str(1))
plt.plot(noise[0])
plt.subplot('42'+str(2))
plt.imshow(images[0])

plt.subplot('42'+str(3))
plt.plot(noise[1])
plt.subplot('42'+str(4))
plt.imshow(images[1])

plt.subplot('42'+str(5))
plt.plot(noise[2])
plt.subplot('42'+str(6))
plt.imshow(images[2])

plt.subplot('42'+str(7))
plt.plot(noise[3])
plt.subplot('42'+str(8))
plt.imshow(images[3])

plt.show()
