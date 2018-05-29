print('Importing necessary packages and modules')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle as pk

#from keras import backend as K
#import os

#def set_keras_backend(backend):

#    if K.backend() != backend:
#        os.environ['KERAS_BACKEND'] = backend
#        reload(K)
#        assert K.backend() == backend

#set_keras_backend("tensorflow")
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, Cropping2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.backend import log

class DCGAN(object):
    def __init__(self, load_state =False, img_rows=600, img_cols=600, channel=1):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.DM = None  # discriminator model
        self.AM = None  # adversarial model
        if load_state:
            print('Loading Previous State')
            try:
                self.load_dcgan()
            except IOError:
                print('Previous state not saved, beginning with fresh state.')

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 64
        dropout = 0.25
        
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth, 5, strides=2, input_shape=input_shape,\
            padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(BatchNormalization(momentum=0.9))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(BatchNormalization(momentum=0.9))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=2, padding='same'))
        self.D.add(BatchNormalization(momentum=0.9))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*16, 5, strides=2, padding='same'))
        self.D.add(BatchNormalization(momentum=0.9))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        depth = 64
        dim1 = 19
        dim2 = 19
        
        self.G.add(Dense(dim1*dim2*depth*16, input_dim=64))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim1, dim2, depth*16)))

        self.G.add(Conv2DTranspose(depth*8, 5, strides = 2, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(depth*4, 5, strides = 2, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        
        self.G.add(Conv2DTranspose(depth*2, 5, strides = 2, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        
        self.G.add(Conv2DTranspose(depth, 5, strides = 2, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(1, 5, strides = 2, padding='same'))
        self.G.add(Cropping2D(cropping=((4,4),(4,4))))
        self.G.add(Activation('tanh'))
        self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = Adam(lr=0.0001,beta_1=0.5, decay=6e-10)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = Adam(lr=0.0002,beta_1=0.5, decay=6e-10)
        self.AM = Sequential()
        self.AM.add(self.generator())
        discriminator =self.discriminator_model()
        discriminator.trainable=False
        self.AM.add(discriminator)
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.AM

    def save_dcgan(self):
        model_type = ['D', 'G', 'DM','AM']
        for m in model_type:
            model = getattr(self, m)
            # serialize model to JSON
            with open(m+".json", "w") as f: f.write(model.to_json())
            # serialize weights to HDF5
            model.save_weights(m+"_weights.h5")
            
    def load_dcgan(self):
        model_type = ['D', 'G', 'DM','AM']
        for m in model_type:
            model = getattr(self, m)
            # load json and create model
            with open(m+'.json', 'r') as f: model = model_from_json(f.read())
            # load weights into new model
            model.load_weights(m+"_weights.h5")
            
class DustDCGAN(object):
    def __init__(self,data,test=False,load_state=False):
        
        if not test:
            print('Loading Data')
            #load list of dust maps
            dust_maps=[]
            with open(data,'rb') as f:
                while True:
                    try:
                        dust_maps.extend(pk.load(f))
                    except EOFError:
                        break
        
            self.img_rows,self.img_cols = np.shape(dust_maps[0]) 
            self.channel = 1
        
            #normalize dust maps across entire set
            dmean = np.mean(dust_maps)
            dstd = np.std(dust_maps)
            self.x_train = (dust_maps - dmean)/dstd
        
            # don't need the unormalized maps
            del dust_maps
        
            #format the training data, for 2d images keras expects one dim to be the num of channels
            #first dim is number of training samples, then image shape, then channels
            self.x_train=np.array(self.x_train)
            self.x_train = self.x_train.reshape(-1, self.img_rows,\
            self.img_cols, 1).astype(np.float32)
            
            ##initialize the discriminator, adversarial models and the generator
            self.DCGAN = DCGAN(load_state=load_state, img_rows=self.img_rows, img_cols=self.img_cols)
            self.generator = self.DCGAN.generator()
            self.discriminator =  self.DCGAN.discriminator_model()
            self.adversarial = self.DCGAN.adversarial_model()
            
        else:
            ##initialize the discriminator, adversarial models and the generator
            self.DCGAN = DCGAN(load_state=load_state, img_rows=600, img_cols=600)
            self.generator = self.DCGAN.generator()
            self.discriminator =  self.DCGAN.discriminator_model()
            self.adversarial = self.DCGAN.adversarial_model()
            
            print('Summary')
            print(self.discriminator.summary())
            print(self.adversarial.summary())

    def train(self, train_steps=2000, batch_size=32, save_interval=0):
        print('Training Beginning')
        for i in range(train_steps):
            # First train the discriminator with correct labels
            # Randomly select batch from training samples
            images_train = self.x_train[np.random.randint(0,
                self.x_train.shape[0], size=batch_size), :, :, :]
            # Generate fake images from generator
            noise = np.random.normal(loc=0., scale=1., size=[batch_size, 64])
            images_fake = self.generator.predict(noise)
            # Combine true and false sets with correct labels and train discriminator
            y = np.random.binomial(1,.99,size=[batch_size, 1])
            d_loss_real = self.discriminator.train_on_batch(images_train, y)
            y =np.random.binomial(1,.01,size=[batch_size, 1])
            d_loss_fake = self.discriminator.train_on_batch(images_fake,y)
            d_loss = np.add(d_loss_fake,d_loss_real)/2
            # Now train the adversarial network
            # Create new fake images labels as if they are from the training set
            y = np.ones([batch_size, 1])
            noise = np.random.normal(loc=0., scale=1., size=[batch_size, 64])
            a_loss = self.adversarial.train_on_batch(noise, y)
            # Generate log messages
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            if i%10==0:
                print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.DCGAN.save_dcgan()
                    noise_input = np.random.normal(loc=0., scale=1., size=[16, 64])
                    filename = "Dust_sims_%d.png" % (i+1)
                    self.plot_images(filename=filename, samples=noise_input.shape[0],noise=noise_input)

    def plot_images(self, filename=None, fake=True, samples=16, noise=None):
        if fake:
            if noise is None:
                noise = np.random.normal(loc=0., scale=1., size=[samples, 64])
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='viridis')
            plt.axis('off')
        plt.tight_layout()
        if filename!=None:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

            
    
