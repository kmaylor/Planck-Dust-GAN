print('Importing DCGAN Class')
from dust_gan import DustDCGAN
print('Instantiate GAN')
dust_gan = DustDCGAN('/home/kmaylor/Cut_Maps/Planck_dust_cuts_353GHz_norm_log_2.h5',test=False,load_dir=None)
print('Begin Training')
dust_gan.train(train_steps=24000, batch_size=32, save_interval=500)
print('Saving GAN')
dust_gan.KGAN.save_state()

