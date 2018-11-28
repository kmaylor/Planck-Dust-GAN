print('Importing DCGAN Class')
from dust_gan import DustDCGAN
print('Instantiate GAN')
dust_gan = DustDCGAN('/home/kmaylor/Cut_Maps/Planck_dust_cuts_353GHz_norm_log_2.h5',test=False,save_dir='/home/kmaylor/Saved_Models/Model_8.4.8.4.8.4.4.2',load_dir=None)
print('Begin Training')
dust_gan.train(train_steps=15000, batch_size=16, save_interval=500)
print('Saving GAN')
dust_gan.KGAN.save_state()

