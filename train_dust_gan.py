from dust_gan import DustDCGAN

dust_gan = DustDCGAN('/global/homes/k/kmaylor/cori/Maps_and_Makers/Planck_dust_cuts_353GHz.pk',load_state=True)
dust_gan.train(train_steps=2000, batch_size=32, save_interval=500)
dust_gan.DCGAN.save_dcgan()
#dust_gan.plot_images(fake=True)
#dust_gan.plot_images(fake=False, save2file=True)
