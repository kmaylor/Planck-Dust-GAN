import numpy as np
import pickle as pk
from kgan import KGAN




class DustDCGAN(object):
    def __init__(self,data,test=False,load_state=False):
        
        kernels = [10,8,4,2]
        strides = [5,4,2,1]#[8,5,3,1]

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
            self.KGAN = KGAN(img_rows=self.img_rows, img_cols=self.img_cols, load_dir=None)
            self.KGAN.strides = strides
            self.KGAN.kernels = kernels
            self.KGAN.depth_scale = [8,4,2,1][::-1]
            
        else:
            ##initialize the discriminator, adversarial models and the generator
            self.KGAN = KGAN(img_rows=600, img_cols=600, load_dir=None)
            self.KGAN.strides = strides
            self.KGAN.kernels = kernels
            self.KGAN.depth_scale = [2,2,2,2,1][::-1]
            print('Summary')
            self.KGAN.discriminator()
            self.KGAN.generator()
            self.KGAN.discriminator_model()
            self.KGAN.adversarial_model()
            print('AM',self.KGAN.get_model_memory_usage(16,'AM'))
            print('DM',self.KGAN.get_model_memory_usage(16,'DM'))

    def train(self, train_steps=2000, save_interval=100, verbose = 10, batch_size=16):
        self.KGAN.train(self.x_train, 'Gen_images/Dust_sims',train_steps=train_steps,
         batch_size=batch_size, save_interval=save_interval, verbose = verbose)

            
    
