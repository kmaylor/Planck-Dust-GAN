print('Importing necessary packages and modules')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle as pk

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, Cropping2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.backend import log, count_params
from keras.initializers import Zeros, Constant

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
            try:
            	print('Loading Previous State')
            	self.load_dcgan()
            except IOError:
                print('Previous state not saved, beginning with fresh state.')

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential(name='Discriminator')
        depth = 64
        dropout = 0.25
        
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth, 20, strides=8, input_shape=input_shape,\
            padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 10, strides=4, padding='same'))
        self.D.add(BatchNormalization(momentum=0.9))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(BatchNormalization(momentum=0.9))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        #self.D.add(Conv2D(depth*8, 5, strides=2, padding='same'))
        #self.D.add(BatchNormalization(momentum=0.9))
        #self.D.add(LeakyReLU(alpha=0.2))
        #self.D.add(Dropout(dropout))

        #self.D.add(Conv2D(depth*16, 5, strides=2, padding='same'))
        #self.D.add(BatchNormalization(momentum=0.9))
        #self.D.add(LeakyReLU(alpha=0.2))
        #self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential(name='Generator')
        depth = 64
        dim1 = 10
        dim2 = 10
        
        self.G.add(Dense(dim1*dim2*depth*4, input_dim=64, kernel_initializer=Zeros(),bias_initializer=Constant(value=0.1)))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim1, dim2, depth*4)))

        #self.G.add(Conv2DTranspose(depth*8, 5, strides = 2, padding='same'))
        #self.G.add(BatchNormalization(momentum=0.9))
        #self.G.add(Activation('relu'))

        #self.G.add(Conv2DTranspose(depth*4, 5, strides = 2, padding='same'))
        #self.G.add(BatchNormalization(momentum=0.9))
        #self.G.add(Activation('relu'))
        
        self.G.add(Conv2DTranspose(depth*2, 5, strides = 2, padding='same', kernel_initializer=Zeros(),bias_initializer=Constant(value=0.1)))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        
        self.G.add(Conv2DTranspose(depth, 10, strides = 4, padding='same', kernel_initializer=Zeros(),bias_initializer=Constant(value=0.1)))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(1, 20, strides = 8, padding='same', kernel_initializer=Zeros(),bias_initializer=Constant(value=0.1)))
        self.G.add(Cropping2D(cropping=((20,20),(20,20))))
        self.G.add(Activation('tanh'))
        self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = Adam(lr=0.0001,beta_1=0.5, decay=0)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = Adam(lr=0.0002,beta_1=0.5, decay=0)
        self.AM = Sequential()
        self.AM.add(self.generator())
        discriminator =self.discriminator()
        for layer in discriminator.layers:
        	layer.trainable=False
        self.AM.add(discriminator)
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.AM

    def save_dcgan(self):
    	model_type = ['D', 'G', 'DM','AM']
    	for m in model_type:
            getattr(self, m).save(m[0]+"_model.h5")
                        
    def load_dcgan(self):
    	model_type = ['D', 'G', 'DM','AM']
    	for m in model_type:
            setattr(self, m, load_model(m[0]+"_model.h5"))
        
    
    def get_model_memory_usage(batch_size, model):

        shapes_mem_count = 0
        for l in model.layers:
            single_layer_mem = 1
            for s in l.output_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem

        trainable_count = np.sum([count_params(p) for p in set(model.trainable_weights)])
        non_trainable_count = np.sum([count_params(p) for p in set(model.non_trainable_weights)])

        total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
        gbytes = np.round(total_memory / (1024.0 ** 3), 3)
        return gbytes


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
            #x = np.concatenate((images_train, images_fake))
            y = np.random.binomial(1,.95,size=[batch_size, 1])
            #y[batch_size:, :] =np.random.binomial(1,.1,size=[batch_size, 1])

            d_loss_real = self.discriminator.train_on_batch(images_train, y)
            y =np.random.binomial(1,.05,size=[batch_size, 1])
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
            if i%100==0:
                print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.DCGAN.save_dcgan()
                    filename = "Dust_sims_%d.png" % (i+1)
                    self.plot_images(filename=filename,noise =noise[:8], samples=16)

    def plot_images(self, filename=None, fake=True, samples=16, noise=None):
        #if fake:
        if noise is None:
                noise = np.random.normal(loc=0., scale=1., size=[int(samples/2), 64])
        images = self.generator.predict(noise)
        #else:
        i = np.random.randint(0, self.x_train.shape[0], int(samples/2))
        images = np.concatenate((images,self.x_train[i, :, :, :]))
        preds = self.discriminator.predict(images)
        #print(preds)
        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='viridis')
            plt.ylabel(preds[i])
            plt.axis('off')
        plt.tight_layout()
        if filename!=None:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

            
    
