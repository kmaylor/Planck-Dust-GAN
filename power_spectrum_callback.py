from scipy import fftpack
import numpy as np
from ImageTools import getInterpolatedPixelValues


class PSDCallback(object):
    
    def __init__(self,data_path):

        with h5py.File(data_path, 'r') as hf:
            real_images=np.array([i for i in hf.values()])
        real_ps = []
        for image in real_images:
            real_ps.append(self.calcSquareImagePSD(image))
        self.average_real_ps = np.mean(np.array(real_ps),axis=0)
        self.neg_log_like = 1e30
        
    def _azimuthalAverage(self,image):
        averages = np.array([])
        center = (image.shape - np.array([1.0, 1.0])) / 2.0
        # Take half of the distance from the center to the closest edge
        max_radius = int(min(image.shape) / 2.0 - 1.0)
        # Loop over the radius bins starting from smallest (zero). Each bin is size 1.
        for radius in range(0, max_radius+1):
            # The number of pixels averaged is equal to the circumference+1 of this radius bin
            circumference = 2.0*np.pi*radius
            thetas = np.linspace(0.0, 2*np.pi, circumference+1, False)
            xcoords = radius*np.cos(thetas) + center[1]
            ycoords = radius*np.sin(thetas) + center[0]
            values = getInterpolatedPixelValues(image, xcoords, ycoords)
            avg = np.sum(values) / values.size
            averages = np.append(averages, avg)
        return averages

    """
    Calculate the 1d & 2d power spectral density of a square image
    1d psd is calculated from 2d psd by starting at the center and averaging radially until the nearest edge
    image must be square
    """
    def calcSquareImagePSD(self,image):
        # Get the 2D FFT of the image
        # Then shift it so that the DC component is at the center
        fft2d = fftpack.fftshift( fftpack.fft2(image) )
    
        # Square the magnitude to get the 2D PSD
        psd2d = np.abs(fft2d)**2
    
        # Average azimuthally with ever-increasing radii starting at the center and towards the corners
        # This averages in a circle at the center of the image
        psd1d = self._azimuthalAverage(psd2d)
    
        return psd1d
    
    def gen_images(gan):
        noise = np.random.normal(loc=0., scale=1., size=[100, gan.latent_dim])
        return gan.models['generator'].predict(noise)
    
    def __call__(self, gan):
        
        fake_images = self.gen_images(gan)
        fake_ps = []
        for image in fake_images:
            real_ps.append(self.calcSquareImagePSD(image))
        average_fake_ps = np.mean(np.array(fake_ps),axis=0)
        fake_ps_cov = np.cov(np.array(fake_ps))
        cho_cov = cho_factor(fake_ps_cov)
        diff = average_fake_ps-self.average_real_ps
        chisq = dot(diff,cho_solve(cho_cov,diff))
        det = np.linalg.det(fake_ps_cov)
        new_neg_log_like = chisq-np.log(det)
        self.neg_log_like = np.min([new_neg_log_like,neg_log_like])
        print('The Negative LogLikelihood is %d'%d(new_neg_log_like))
        if new_neg_log_like == self.neg_log_like:
            if not os.path.exists(str(gan.save_dir)): os.makedirs(str(gan.save_dir))
                for k in ['discriminator','generator']:
                    gan.models[k].save(gan.save_dir+'/'+k+'_best_psd.h5')
        
    
    
