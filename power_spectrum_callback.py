from scipy import fftpack
import numpy as np
from scipy.linalg import cho_factor, cho_solve, inv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class PSDCallback(object):
    
    def __init__(self,real_images):

        #with h5py.File(data_path, 'r') as hf:
        #    real_images=np.array([i for i in hf.values()])
        real_ps = []
        for image in real_images[:,:,:,0]:
            real_ps.append(self.calcSquareImagePSD(image))
        self.average_real_ps = np.mean(np.array(real_ps),axis=0)
        self.neg_log_like = 1e30

    def getInterpolatedPixelValues(self,image, x, y):
	    x = np.asarray(x)
	    y = np.asarray(y)
	
	    x0 = np.floor(x).astype(int)
	    x1 = x0 + 1
	    y0 = np.floor(y).astype(int)
	    y1 = y0 + 1
	
	    x0 = np.clip(x0, 0, image.shape[1]-1)
	    x1 = np.clip(x1, 0, image.shape[1]-1)
	    y0 = np.clip(y0, 0, image.shape[0]-1)
	    y1 = np.clip(y1, 0, image.shape[0]-1)
	
	    center = image[y0, x0]
	    top = image[y1, x0]
	    right = image[y0, x1]
	    topright = image[y1, x1]
	
	    w_center = (x1-x) * (y1-y)
	    w_top = (x1-x) * (y-y0)
	    w_right = (x-x0) * (y1-y)
	    w_topright = (x-x0) * (y-y0)
	
	    return w_center*center + w_top*top + w_right*right + w_topright*topright
    
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
            values = self.getInterpolatedPixelValues(image, xcoords, ycoords)
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
    
    def gen_images(self,gan):
        noise = np.random.normal(loc=0., scale=1., size=[100, gan.latent_dim])
        return gan.models['generator'].predict(noise)
    
    def __call__(self, gan):
        
        fake_images = self.gen_images(gan)[:,:,:,0]
        fake_ps = []
        for image in fake_images:
            fake_ps.append(self.calcSquareImagePSD(image))
        average_fake_ps = np.mean(np.array(fake_ps),axis=0)
        fake_ps_cov = np.cov(np.array(fake_ps),rowvar=False)
        diff = average_fake_ps-self.average_real_ps
        chisq = sum(diff**2/np.diag(fake_ps_cov))
        new_neg_log_like = chisq+np.sum(np.log(np.diag(fake_ps_cov)))
        self.neg_log_like = np.min([new_neg_log_like,self.neg_log_like])
        print('The Negative LogLikelihood is %f' % (new_neg_log_like))
        if new_neg_log_like == self.neg_log_like:
            if not os.path.exists(str(gan.save_dir)): os.makedirs(str(gan.save_dir))
            for k in ['discriminator','generator']:
                gan.models[k].save(gan.save_dir+'/'+k+'_best_psd.h5')
        
    
    
