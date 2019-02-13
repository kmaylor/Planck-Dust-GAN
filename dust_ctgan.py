import numpy as np
import h5py
from KGAN.gans.ct_gan import CTGAN as GAN



     
        #kernels = [4,4,4,4]
        #strides =
depth=32
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


dustGAN = GAN((img_rows,img_cols,channel),
                        strides=strides,
                        kernels=kernels,
                        load_dir=None,
                        min_depth=depth,
                        save_dir = '/home/kmaylor/Saved_Models/CT_GAN')
            
         
dustGAN.train(x_train,
              '/home/kmaylor/Gen_images/CT_GAN_figures/dust',
              train_steps=50000,
              batch_size=32,
              save_rate=500,
              mesg_rate = 100, 
              train_rate=(5,1),
             )
