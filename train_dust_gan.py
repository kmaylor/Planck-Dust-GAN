from dust_gan import DustDCGAN
print('Instantiate GAN')
dust_gan = DustDCGAN('D:\Projects\Maps_and_Makers\Planck_dust_cuts_353GHz.pk',test=False,load_state=True)
print('Begin Training')
dust_gan.train(train_steps=12000, batch_size=32, save_interval=100)
print('Saving GAN')
dust_gan.DCGAN.save_dcgan()

