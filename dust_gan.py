
import numpy as np
import h5py
from KGAN.gans.dcgan import DCGAN as GAN

from power_spectrum_callback import PSDCallback



     
        #kernels = [4,4,4,4]
        #strides =
depth=64
        #kernels = [10,4,4]
        #strides = [10,4,2]
kernels = [5,5,5,5]
strides = [2,2,2,2]
data = '/home/kmaylor/Cut_Maps/Planck_dust_cuts_353GHz_norm_log_res256.h5'

img_rows = 256
img_cols = 256
channel = 1
        


with h5py.File(data, 'r') as hf:
    x_train=np.array([i for i in hf.values()]).reshape(-1, img_rows,img_cols, 1).astype(np.float32)

call_back = PSDCallback(x_train,statistic='ps')

dustGAN = GAN((img_rows,img_cols,channel),
                        strides=strides,
                        kernels=kernels,
                        load_dir='/home/kmaylor/Saved_Models/DCGAN128besthistx1',
                        min_depth=depth,
                        latent_dim = 64,
                        save_dir = '/home/kmaylor/Saved_Models/DCGAN64bestpsd')
            
         
dustGAN.train(x_train,
              '/home/kmaylor/Gen_images/DCGAN_figures/dust2',
              train_steps=10000,
              batch_size=64,
              save_rate=2000,
              mesg_rate = 1, 
              train_rate=(1,1),
              call_back = call_back )
