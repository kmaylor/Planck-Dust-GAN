print('Importing DCGAN Class')
from dust_gan import DustDCGAN
print('Instantiate GAN')
dust_gan = DustDCGAN('/home/kmaylor/Cut_Maps/Planck_dust_cuts_353GHz_norm_log_2.h5',test=False,save_dir='/home/kmaylor/Saved_Models/Model_55.44.42.42_lr0002_72000',load_dir='/home/kmaylor/Saved_Models/Model_55.44.42.42_lr0002_48000')
print('Begin Training')
dust_gan.train(train_steps=24000, batch_size=32, save_interval=500)
print('Saving GAN')
dust_gan.KGAN.save_state()

