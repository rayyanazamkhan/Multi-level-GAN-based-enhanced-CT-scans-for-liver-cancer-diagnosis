import numpy as np
import os
import random
import skimage.transform as trans
from keras.models import *
from keras.losses import *
from keras.layers import *
from keras.applications import imagenet_utils
from keras.optimizers import *
from keras.callbacks import *
from keras.utils import *
from keras.backend.common import normalize_data_format
from keras.objectives import binary_crossentropy
from keras import backend as K
import tensorflow as tf
import tensorlayer as tl
from keras.preprocessing.image import ImageDataGenerator
import cv2
import skimage.io as io
import natsort
from tensorflow.python.ops import io_ops
import math
from PIL import Image, ImageEnhance
#import tensorflow_addons as tfa
from keras.regularizers import l2
#from utils import load_images
###############################Parameters######################################
# Data & model configuration
image_size=(128,128)
num_channels = 3
batch_size = 5
epoch_num=50
train_images=18473
test_images=1000
batches = int(train_images/batch_size)
Training = False     #True= If want to train on new dataset
critic_updates =3
No_prior_weights = False #True= if any of discriminative/generative model weights are not available otherwise false
Train_sharp_image_path = '/home/rak038/Enhancement/Train/1/0/'           #Sharp
Train_blur_image_path = '/home/rak038/Enhancement/Train/0/0/'            #Blur
Test_image_path = '/home/rak038/Enhancement/Train/0/0/'
Test_save_path_1 = "/home/rak038/Enhancement/Test/Out_GAN/"

################################# Models #################################
def generative(pretrained_weights_gen,input_shape):
    
    Im_128 = Input(shape=input_shape, name='Generator_Input')
    Im_64 = MaxPooling2D(pool_size=(2, 2))(Im_128)
    Im_32 = MaxPooling2D(pool_size=(4, 4))(Im_128)
    
    Out1=enhancement(Im_32,Im_32)
    Out1_up=UpSampling2D(size = (2,2))(Out1)
    Con1=Concatenate(axis=3)([Im_64,Out1_up])
    Out2=enhancement(Con1,Im_64)
    Out2_up=UpSampling2D(size = (2,2))(Out2)
    Con2=Concatenate(axis=3)([Im_128,Out2_up])
    Out3=enhancement(Con2,Im_128)



    model = Model(inputs=Im_128, outputs=Out3, name='Generator')
    
    if(pretrained_weights_gen):
        model.load_weights(pretrained_weights_gen)
    return model
    
################################################################
def enhancement(Inputs0,Inputs1):
    conv1 = Conv2D(64, 7, activation = 'relu', padding = 'same', dilation_rate=(1,1))(Inputs0)
    conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', dilation_rate=(2,2))(conv1)
    conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv2 = BatchNormalization()(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=(4,4))(conv2)
    conv3 = BatchNormalization()(conv3)

    conv4 = Conv2DTranspose(128, 3, activation = 'relu', padding = 'same', dilation_rate=(2,2))(UpSampling2D(size = (2,2))(conv3))
    conv4 = BatchNormalization()(conv4)
    conv5 = Conv2DTranspose(64, 3, activation = 'relu', padding = 'same', dilation_rate=(4,4))(UpSampling2D(size = (2,2))(conv4))
    conv5 = BatchNormalization()(conv5)
    conv6 =  Conv2D(3, 7, activation = 'tanh', padding = 'same', dilation_rate=(1,1))(conv5)
    Outputs = Add()([conv6,Inputs1])
    Outputs = Lambda(lambda z: z/2)(Outputs)
    return (Outputs)


def discriminator(pretrained_weights_dis,input_shape):
    
    Inputs = Input(shape=input_shape, name='Discriminator_Input')
    conv2 = Conv2D(64, 4, padding = 'same', strides=(2,2))(Inputs)
    conv2 = LeakyReLU(0.2)(conv2)
    conv3 = Conv2D(128, 4, padding = 'same', strides=(2,2))(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(0.2)(conv3)
    conv4 = Conv2D(256, 4, padding = 'same', strides=(2,2))(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(0.2)(conv4)
    conv5 = Conv2D(512, 4, padding = 'same', dilation_rate=(2,2))(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(0.2)(conv5)
    conv6 = Conv2D(3, 4, padding = 'same', dilation_rate=(1,1))(conv5)
    x = Flatten()(conv6)
    x = Dense(2048, activation='tanh')(x)
    Outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=Inputs, outputs=Outputs, name='Discriminator')
         
    if(pretrained_weights_dis):
        model.load_weights(pretrained_weights_dis)
    return model




def generator_containing_discriminator_multiple_outputs(generator, discriminator):
    inputs = Input(shape = (image_size[0],image_size[1],num_channels))
    discriminator.trainable=False
    generator.trainable=True
    generated_image = generator(inputs)
    outputs = discriminator(generated_image)
    model = Model(inputs=inputs, outputs=[generated_image, outputs])
    return model



#############################VGG16 Model###########################################

def VGG16(include_top=False ,weights='imagenet', classes=1000,
        input_shape = [image_size[0],image_size[1],num_channels],
        classifier_activation='softmax'):
    # if include_top=false input must be (224,224,3)
  # Determine proper input shape

  inputs = Input(shape=input_shape)
  # Block 1
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

  # Block 2
  x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
  x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

  # Block 3
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

  # Block 4
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

  # Block 5
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

  if include_top:
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)

    imagenet_utils.validate_activation(classifier_activation, weights)
    x = Dense(classes, activation=classifier_activation,name='predictions')(x)

  model = Model(inputs, x, name='vgg16')

  # Load weights.
  if weights == 'imagenet':
    if include_top:
      model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    else:
      model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
  return model
def preprocess_input(x, data_format=None):
  return imagenet_utils.preprocess_input(x, data_format=data_format, mode='caffe')


#################################Losses#########################################
#= 3
#loss_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(image_size[0],image_size[1],num_channels))
#loss_model.trainable = False
#dense_loss= K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

#Load VGG model
vgg = VGG16()
vgg.trainable = False
lossModel = Model([vgg.input], vgg.get_layer('block1_conv1').output, name = 'vggL')
def ENTROPY(y_true,y_pred):
    ent = sparse_categorical_crossentropy(y_true, y_pred)
    res = MAE(ent[0],ent[1])
    return res
#@tf.function()
def perceptual_loss(y_true, y_pred):
    l1_loss = MAE(lossModel(y_true),lossModel(y_pred))
    #ent_loss = ENTROPY(y_true,y_pred)
    return l1_loss

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

optimizer =  RMSprop(lr=1E-5, epsilon=1e-08)#Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
###################################Data Processing#############################
def preprocess_image(img):
    # Preprocess the image file
    img = cv2.resize(img,image_size)
    img = (img - 127.5) / 127.5
    return img

def deprocess_image(img):
    # Deprocess the image file
    img = img * 127.5 + 127.5
    return img.astype('uint8')

def list_files(directory):
    # Lists all the image files in a directory
    files = natsort.natsorted(os.listdir(directory))
    return files
def load_file(path):
    # Load an image file from a given path
    im = cv2.imread(path)
    return im

def load_images(path_sharp, path_blur, n_images):
    # Load n_images from a given path randomly
    all_A_paths, all_B_paths = list_files(path_sharp), list_files(path_blur)
    zipped_lists = list(zip(all_A_paths, all_B_paths))
    random.shuffle(zipped_lists)
    all_A_paths, all_B_paths = zip(*zipped_lists)
    A, B = [], []
    A_paths, B_paths = [], []
    for path_A, path_B in zip(all_A_paths, all_B_paths):
        A_im, B_im = load_file(path_sharp+path_A), load_file(path_blur+path_B)
        A.append(preprocess_image(A_im))
        B.append(preprocess_image(B_im))
        A_paths.append(path_sharp+path_A)
        B_paths.append(path_blur+path_B)
        if len(A) > n_images - 1:
            break
    return {
        'A_': np.array(A),
        'A__paths': np.array(A_paths),
        'B_': np.array(B),
        'B__paths': np.array(B_paths)
    }
datagen = ImageDataGenerator(rescale = None,
                            featurewise_center=False,
                            featurewise_std_normalization=False,
                            rotation_range=5,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            shear_range=0.1,
                            zoom_range=0.1,
                            horizontal_flip=True,
                            fill_mode='nearest',
                            zca_whitening=False)
########################## Test Output Images #################################
def testGenerator(test_path,target_size):
    ls=natsort.natsorted(os.listdir(test_path))
    for image_file_name in ls:
        if image_file_name.endswith(".tif"):
            im = cv2.imread(test_path+image_file_name)
            img = preprocess_image(im)
            #img=cv2.resize(im,target_size)
            img = np.reshape(img,(1,)+img.shape)
            yield img

def saveResult(save_path,npyfile):
    for i,item in enumerate(npyfile):
        im = deprocess_image(item)
        cv2.imwrite(os.path.join(save_path ,'%d.tif'%i),im)
###################################Train & Validate############################
'''Train and Validate first discriminator then Generative model'''

if Training:

    data = load_images(Train_sharp_image_path, Train_blur_image_path,train_images)
    x_train, y_train = data['B_'], data['A_']
    for iterations in range(epoch_num):
        print('Epoch:',iterations)
        if No_prior_weights:            
            d = discriminator(None,(image_size[0],image_size[1],num_channels))
            g = generative(None,(image_size[0],image_size[1],num_channels))
            No_prior_weights=False
        else:
            d = discriminator('discriminator1.h5',(image_size[0],image_size[1],num_channels))
            g = generative('generative1.h5',(image_size[0],image_size[1],num_channels))
        d.summary()
        g.summary()    
        d_on_g = generator_containing_discriminator_multiple_outputs(g, d)
        d.compile(optimizer = optimizer, loss = wasserstein_loss)
        g.compile(optimizer = optimizer, loss=perceptual_loss, metrics = ['accuracy'])
        d_on_g.compile(optimizer=optimizer, loss = [perceptual_loss,wasserstein_loss],metrics = ['accuracy'])
        out_train_true_batch,out_train_fake_batch = np.ones((batch_size,1)),-np.ones((batch_size, 1))
        permutated_indexes = np.random.permutation(x_train.shape[0])
        d_losses = []
        d_on_g_losses = []
        
        for j in range(batches):
            batch_indexes = permutated_indexes[j*batch_size:(j+1)*batch_size]
            image_blur_batch = x_train[batch_indexes]
            image_full_batch = y_train[batch_indexes]
            generated_images = g.predict(x=image_blur_batch, batch_size=batch_size)
            for _ in range(critic_updates):
                #d_loss_real = d.fit(datagen.flow(x=image_full_batch,y=out_train_true_batch, batch_size=batch_size),
                #                    steps_per_epoch=len(image_full_batch)/batch_size, epochs=1)
                #d_loss_fake = d.fit(datagen.flow(x=generated_images, y=out_train_fake_batch, batch_size=batch_size),
                #                    steps_per_epoch=len(generated_images)/batch_size, epochs=1)
                d_loss_real = d.train_on_batch(image_full_batch, out_train_true_batch)
                d_loss_fake = d.train_on_batch(generated_images, out_train_fake_batch)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                d_losses.append(d_loss)
            print('batch {} d_loss : {}'.format(j+1, np.mean(d_losses)))
            d.trainable = False
            
            d_on_g_loss = d_on_g.train_on_batch(image_blur_batch, [image_full_batch, out_train_true_batch])
            d_on_g_losses.append(d_on_g_loss)
            print('batch {} d_on_g_loss : {}'.format(j+1, np.mean(d_on_g_losses)))
            

            
            g.save_weights('generative1.h5')
            d.save_weights('discriminator1.h5')
            d.trainable =True

g = generative('generative1.h5',(image_size[0],image_size[1],num_channels))
g.summary()
#################################Train Generative Model #######################
#test_generator = testGenerator(test_path = Test_image_path,target_size=image_size)

#gen_images = g.predict_generator(test_generator,test_images,verbose=1)

saveResult(Test_save_path_1,gen_images)
