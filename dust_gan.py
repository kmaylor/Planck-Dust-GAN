import numpy as np
import h5py
from KGAN.kgan import KGAN




class DustDCGAN(object):
    def __init__(self,data,test=False,load_dir=None):
        
        #kernels = [4,4,4,4,2]
        #strides = [4,4,4,2,1]
        kernels = [10,4,4]
        strides = [10,4,2]
        if not test:
            with h5py.File(data, 'r') as hf:
                self.x_train = np.array([i for i in hf.values()]).reshape(-1, 900, 900, 1).astype(np.float32)
            ##initialize the discriminator, adversarial models and the generator
            self.KGAN = KGAN(strides=strides,kernels=kernels,img_rows=900,
             img_cols=900, load_dir=None, gpus = 2)

            
            #self.KGAN.depth_scale = [6,4,2,1][::-1]
            
        else:
            ##initialize the discriminator, adversarial models and the generator
            self.KGAN = KGAN(img_rows=600, img_cols=600, load_dir=None)
            self.KGAN.strides = strides
            self.KGAN.kernels = kernels
            self.KGAN.depth_scale = [2,2,2,1][::-1]
            print('Summary')
            self.KGAN.discriminator()
            self.KGAN.generator()
            self.KGAN.discriminator_model()
            self.KGAN.adversarial_model()
            print('AM',self.KGAN.get_model_memory_usage(16,'AM'))
            print('DM',self.KGAN.get_model_memory_usage(16,'DM'))

    def train(self, train_steps=2000, save_interval=100, verbose = 10, batch_size=32):
        self.KGAN.train(self.x_train, 'Gen_images/Dust_sims',train_steps=train_steps,
         batch_size=batch_size, save_interval=save_interval, verbose = verbose)

            
    def gen_batch(batch_size):
        batch=[]
        with h5py.File('Planck_dust_cuts_353GHz_norm_log.h5', 'r') as hf:
            for i in range(batch_size):
                group = np.random.randint(0,len(hf.keys()))
                map_group =  hf.get(str(group))
                batch.append(map_group[np.random.randint(0,len(map_group))])
        return np.array(batch).reshape(-1, self.img_rows, self.img_cols, 1).astype(np.float32)
