# Create a multi-band CEERS catalog
# Detection band is a weighted co-add of F200W + F277W + F356W

__author__ = "Henry C. Ferguson, STScI"
__version__ = "1.4.0"
__license__ = "BSD3"

# Version notes
# 1.0.1 -- save the coadd file
# 1.3.0 -- Rewritten for CEERS v0.1
# 1.4.0 -- Rewritten for CEERS dr0.5; extra kron fluxes and flux radii routines; revised read routines

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, QTable, vstack
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import units as u
from astropy import stats as astrostats
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft
from scipy.ndimage import binary_dilation, generate_binary_structure
from astropy.nddata import NDData, StdDevUncertainty, InverseVariance, Cutout2D
import sys
from os import path
from dataclasses import dataclass, asdict
import bz2
from importlib_metadata import version # For getting versions of dependencies
import dill

# Photutils imports
#import photutils
#print('photutils', photutils.__version__)

from photutils import Background2D, MedianBackground, detect_sources, deblend_sources, SourceCatalog#, source_properties (new API)
from photutils.utils import calc_total_error

def compressed_dill(data,outputdir,outputfile):
    output_path = path.join(outputdir,outputfile+".pbz2")
    with bz2.BZ2File(output_path, 'wb') as f:
       dill.dump(data, f)

@dataclass
class PhotCat:
    ab_zpt: float = 31.4 # 1 nJy in ABMag
    coadd_bands: list = ('f277w', 'f356w')
    hst_bands: list = ('f606w','f814w','f125w','f140w','f160w')
    jwst_bands: list = ('f115w','f150w', 'f200w', 'f277w', 'f356w', 'f410m', 'f444w')
    reference_band: str = 'f200w'
    detect_kernel_fwhm: float = 4.0
    detect_threshold_nsigma: float = 3.0
    detect_npixels: int = 4
    detect_kernel_npix: int = 7 
    deblend_npixels: int = 1
    deblend_nlevels: int = 32
    deblend_contrast: float = 0.001
    near_border_kernelsize: int = 21
    flag_isolated_buffer: int = 4

    def __post_init__(self):
        # ACS from https://iopscience.iop.org/article/10.1088/0067-0049/197/2/36
        # WFC3 From  https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/ir-photometric-calibration
        self.hst_zeropoints = {
              'f606w': 26.49,
              'f814w': 25.94,
              'f105w': 26.264,
              'f125w': 26.232,
              'f140w': 26.450,
              'f160w': 25.936
        }
        # Data structures
        self.images = {}
        self.photcats = {}
        self.photometries = {}
        self.kernelfiles = {}
        self.matched_images = {}
        self.matched_photcats = {}
        self.matched_photometries = {}
        self.matching_kernels = {}
        self.matching = {}
        # Convert to nJy for consistency with JWST
        for b in self.hst_zeropoints:
            self.hst_zeropoints[b]  = (self.hst_zeropoints[b]*u.ABmag).to(u.nJy)
        self.populate_catalog_metadata() # Don't do this after loading the images, or they will all be in the metadata!

    def read_jwst(self,ceerspaths,field):
        ''' paths is a dictionary of paths to all the files,
            e.g. ceerspaths[1] from dr05paths.py 
        '''
    # Read in to NDData structures
        self.field = field
        paths = ceerspaths[field]
        maxerr = 1.e6
        for b in self.jwst_bands:
            meta = {}
            bksub_ext = paths['bksub_ext'][b]
            rms_ext = paths['rms_ext'][b]
            mask_ext = paths['tiermask_ext'][b]
            with fits.open(paths['fitsfile'][b]) as hdu:
                wcs = WCS(hdu[bksub_ext].header)
                sci = hdu[bksub_ext].data
                mask = np.bitwise_and(hdu[mask_ext].data.astype(np.int32),1) == 1 # Mask regions of the detector
                pixel_area_sr = hdu[bksub_ext].header['PIXAR_SR']
                nJy_conversion = 1*u.MJy.to(u.nJy)*pixel_area_sr
                # Put the sci array into nJy per pixel
                sci *= nJy_conversion
                meta['PHOTMJSR'] = hdu[bksub_ext].header['PHOTMJSR']
                meta['PIXAR_SR'] = hdu[bksub_ext].header['PIXAR_SR']
                err = hdu[rms_ext].data
                err *= nJy_conversion # convert to nJy per pixel
                err_floor = np.percentile(err[~mask],99) # Set the pixels with 0 error to the 99th percentile
                err_floored = np.choose(err == 0., (err,err_floor))
                err_floored = np.choose(np.isnan(err), (err_floored, maxerr))
                self.images[b] = NDData(sci*u.nJy,
                        InverseVariance(1/(err_floored)**2),
                        mask=mask, wcs=wcs, meta=meta)
        with fits.open(paths['detmap']) as hdu:
            self.detmap = hdu[0].data

    def create_matched(self,kernelpaths):
        ''' Convolve images with a kernel to one reference band '''
        for b in kernelpaths:
            i = kernelpaths[b].find('_')
            j = kernelpaths[b].rfind('_')
            self.kernelfiles[b] = path.basename(kernelpaths[b])
            self.matching[b] = (kernelpaths[b][i+1:i+6], # Convolve from this band
                                kernelpaths[b][j+1:j+6]) # To this one
            #print(i,j,b,self.matching[b])
            with fits.open(kernelpaths[b]) as hdu:
                self.matching_kernels[b] = hdu[0].data
        for b in kernelpaths:
            if b in self.images:
                self.matched_images[b] = convolve_fft(self.images[self.matching[b][0]],
                    self.matching_kernels[b],allow_huge=True)

    def read_hst(self,ceerspaths,field):
        ''' paths is a dictionary of paths to all the files,
            e.g. ceerspaths[1] from v07paths.py 
        '''
    # Read in to NDData structures
        self.field = field
        paths = ceerspaths[field]
        maxerr = 1.e6
        for b in self.hst_bands:
            bksub_ext = paths['bksub_ext'][b]
            rms_ext = paths['rms_ext'][b]
            mask_ext = paths['tiermask_ext'][b]
            with fits.open(paths['fitsfile'][b]) as hdu:
                wcs = WCS(hdu[bksub_ext].header)
                sci = hdu[bksub_ext].data
                mask = np.bitwise_and(hdu[mask_ext].data.astype(np.int32),1) == 1 # Mask regions of the detector
                sci = sci * self.hst_zeropoints[b] # Convert to nJy
                err = hdu[rms_ext].data
                meta = {}
                err_floor = np.percentile(err[~mask],99) # Set the pixels with 0 error to the 99th percentile
                err_floored = np.choose(err == 0., (err,err_floor))
                err_floored = np.choose(np.isnan(err), (err_floored, maxerr))
                err_floored = err_floored * self.hst_zeropoints[b] # Convert to nJy
                #print(err_floor, err.max(), err_floored.min(), err_floored.max())
                self.images[b] = NDData(sci,
                        InverseVariance(1/err_floored**2),
                        mask=mask, wcs=wcs, meta=meta)

    def populate_catalog_metadata(self):
       self.catalog_metadata = {}
       self.catalog_metadata['astropy version'] = version('astropy')
       self.catalog_metadata['photutils version'] = version('photutils')
       self.catalog_metadata['make_catalog version'] = __version__
       self.catalog_metadata['make_catalog parameters'] = asdict(self)

    def write_coadd_photcat(self,outputdir=None,prefix="",suffix="",filetype='ecsv',overwrite=True):
       if outputdir is None:
           outputdir = self.datadir
       coadd_filterstring = ""
       for b in self.coadd_bands:
           coadd_filterstring += f"_{b}"
       coadd_filterstring += "_"
       coadd_filename = f"{prefix}detect{coadd_filterstring}{suffix}.{filetype}"
       self.photcat_coadd.meta = self.catalog_metadata
       self.photcat_coadd.write(path.join(outputdir,coadd_filename),overwrite=overwrite)

    def write_photcat(self,outputdir=None,prefix="",suffix="",filetype='ecsv',overwrite=True):
       for b in self.photcats:
           outfile = f"{prefix}{b}{suffix}.{filetype}"
           self.photcats[b].meta = self.catalog_metadata
           for k in self.images[b].meta:
               self.photcats[b].meta[k] = self.images[b].meta[k]
           self.photcats[b].write(path.join(outputdir,outfile),overwrite=overwrite)

    def write_matched_photcat(self,outputdir=None,prefix="",suffix="",filetype='ecsv',overwrite=True):
       for b in self.matched_photcats:
           matchbands = f"{self.matching[b][0]}_{self.matching[b][1]}"
           outfile = f"{prefix}matched_{matchbands}{suffix}.{filetype}"
           self.matched_photcats[b].meta = self.catalog_metadata
           self.matched_photcats[b].meta['kernelfile'] = self.kernelfiles[b]
           for k in self.images[b].meta:
               self.matched_photcats[b].meta[k] = self.images[b].meta[k]
           self.matched_photcats[b].write(path.join(outputdir,outfile),overwrite=overwrite)

    def write_segmentation_image(self,outputdir=None,prefix="",suffix="",overwrite=True):
       if outputdir is None:
           outputdir = self.datadir
       outseg = path.join(outputdir,f"{prefix}seg{suffix}.fits")
       seghdu = fits.PrimaryHDU(self.seg.data,header=self.seg.wcs.to_header())
       seghdu.writeto(outseg,overwrite=overwrite)

    def write_coadd(self,outputdir=None,prefix="",suffix="",overwrite=True):
       if outputdir is None:
           outputdir = self.datadir
       out_coadd = path.join(outputdir,f"{prefix}coadd{suffix}.fits")
       coadd_hdu = fits.PrimaryHDU(self.coadd.data,header=self.coadd.wcs.to_header())
       coadd_hdu.writeto(out_coadd,overwrite=overwrite)

    def dill_coadd_photometry(self,outputdir=None,prefix="",suffix=""):
        out_file=f"{prefix}photom_object_coadd{suffix}"
        compressed_dill(self.photometry_coadd,outputdir, out_file)

    def dill_segmap(self,outputdir=None,prefix="",suffix=""):
        out_file=f"{prefix}seg_object{suffix}"
        compressed_dill(self.segm_deblend,outputdir, out_file)

    def make_coadd(self):
        ''' Weighted mean of the coadds. Assumes the NDData uncertainties are inverse variance '''
        weight_sum = 0
        coadd = 0
        ref = self.images[self.reference_band]
        coadd_mask = np.zeros(self.images[self.coadd_bands[0]].mask.shape,bool)
        for b in self.coadd_bands:
            weight_sum += self.images[b].uncertainty.array
            coadd += self.images[b].data*self.images[b].uncertainty.array
            coadd_mask = coadd_mask & self.images[b].mask
        coadd /= weight_sum
        self.coadd = NDData(coadd,InverseVariance(weight_sum),mask=coadd_mask,wcs=ref.wcs)
        

    def detect(self):
        # Use the error array to set the detection threshold
        err = self.coadd.uncertainty.represent_as(StdDevUncertainty).array # Extract uncertainty as sigma
        threshold = self.detect_threshold_nsigma * err

        # Before detection, smooth image with Gaussian FWHM = 2 pixels
        sigma = self.detect_kernel_fwhm * gaussian_fwhm_to_sigma  
        npix = self.detect_kernel_npix
        kernel = Gaussian2DKernel(sigma, x_size = npix, y_size = npix)
        kernel.normalize()
        self.convolved_coadd = convolve(self.coadd.data,kernel)

        # Detect
        self.segm_detect = detect_sources(self.convolved_coadd, threshold, 
                npixels = self.detect_npixels,
                mask = self.coadd.mask)
        # Deblend
        self.segm_deblend = deblend_sources(self.convolved_coadd, self.segm_detect, 
                npixels = self.deblend_npixels,
                nlevels = self.deblend_nlevels,
                contrast = self.deblend_contrast)
        # Put the segmap into an NDData array so we have a wcs attached
        self.seg = NDData(self.segm_deblend.data,wcs = self.images[self.reference_band].wcs)

    def photometry_on_convolved_detect_image(self):
        err = self.coadd.uncertainty.represent_as(StdDevUncertainty).array # Extract uncertainty as sigma
        self.photometry_coadd = SourceCatalog(
                self.convolved_coadd, 
                self.segm_deblend, 
                wcs = self.images[self.reference_band].wcs,
                background=np.zeros(self.coadd.data.shape,np.float64), 
                error = err)
        self.my_photcolumns = self.photometry_coadd.default_columns
        self.photcat_coadd = self.photometry_coadd.to_table()
        self.nsources = len(self.photcat_coadd)

    def add_extra_columns(self,
            extra_columns=[
                   'kron_radius',
                   'local_background',
                   'gini',
                   'equivalent_radius']):
        ''' Add extra columns after running photometry on the detect inage '''
        self.my_photcolumns += extra_columns
        self.photcat_coadd.to_table(colums=self.my_photcolumns)

    def photometry_bands(self,bands):
        for b in bands:
            img = self.images[b].data
            err = self.images[b].uncertainty.represent_as(StdDevUncertainty).array
            bkg = np.zeros(img.shape,np.float64)
            wcs = self.images[b].wcs
            ab_zpt = self.ab_zpt
            photometry = SourceCatalog(img, self.segm_deblend, 
                    wcs = wcs, 
                    background = bkg, 
                    error = err)
            self.photometries[b] = photometry
            self.photcats[b] = photometry.to_table(self.my_photcolumns)
            self.photcats[b]['kron_mag'] = -2.5*np.log10(self.photcats[b]['kron_flux']) + ab_zpt

    def extra_kron_photometry(self,bands,kron_params=[4.,2.0,2.0],kron_name="kron4_2",overwrite=False):
        for b  in bands:
            _stuff = self.photometries[b].kron_photometry(kron_params,name=kron_name,overwrite=overwrite)
        if f"{kron_name}_flux" not in self.my_photcolumns:
            self.my_photcolumns += [f"{kron_name}_flux"]
            self.my_photcolumns += [f"{kron_name}_mag"]
            self.my_photcolumns += [f"{kron_name}_fluxerr"]
        for b  in bands:
            self.photcats[b] = self.photometries[b].to_table(self.my_photcolumns)
            self.photcats[b][f"{kron_name}_mag"] = -2.5*np.log10(
                    self.photcats[b][f"{kron_name}_flux"]) + ab_zpt

    def extra_kron_matched_photometry(self,bands,kron_params=[4.,2.0,2.0],
            kron_name="kron4_2",overwrite=False):
        for b  in bands:
            _stuff = self.matched_photometries[b].kron_photometry(
                    kron_params,name=kron_name,overwrite=overwrite)
        if f"{kron_name}_flux" not in self.my_photcolumns:
            self.my_photcolumns += [f"{kron_name}_flux"]
            self.my_photcolumns += [f"{kron_name}_mag"]
            self.my_photcolumns += [f"{kron_name}_fluxerr"]
        for b  in bands:
            self.photcats[b] = self.matched_photometries[b].to_table(self.my_photcolumns)
            self.matched_photcats[b][f"{kron_name}_mag"] = -2.5*np.log10(
                    self.matched_photcats[b][f"{kron_name}_flux"]) + ab_zpt

    def fluxfrac_radii(fluxfracs,overwrite=False):
        for ff in fluxfracs:
           name = f"radius_kron{int(ff*100)}"
           if name not in self.my_photcolumns:
               self.my_photcolumns += [name]
           for b in bands:
                _radii = self.photometries[b].fluxfrac_radius(ff,name=name,overwrite=overwrite)
                self.photcats[b] = self.photometries[b].to_table(self.my_photcolumns)

    def photometry_matched_bands(self):
        for b in self.matched_images:
            img = self.matched_images[b].data
            err = self.images[b].uncertainty.represent_as(StdDevUncertainty).array
            bkg = np.zeros(img.shape,np.float64)
            wcs = self.images[b].wcs
            ab_zpt = self.ab_zpt
            photometry = SourceCatalog(img, self.segm_deblend,
                    wcs = wcs,
                    background = bkg,
                    error = err)
            self.matched_photometries[b] = photometry
            self.matched_photcats[b] = photometry.to_table()
            self.matched_photcats[b]['kron_mag'] = -2.5*np.log10(self.matched_photcats[b]['kron_flux']) + ab_zpt

    def identify_detectors(self):
        '''Include the detector bit map at the source centroid in the catalog '''
        self.photcat_coadd['detmap'] = np.zeros(self.nsources, dtype=np.int32)
        for i in range(self.nsources):
            s = self.photcat_coadd[i]
            x = np.int32(s['xcentroid'])
            y = np.int32(s['ycentroid'])
            self.photcat_coadd['detmap'][i] = self.detmap[y,x]

    def flag_near_border(self,bands):
        '''Make a mask by dilating the region of the detector that was NaNs in the err array.
           If the centroid of a source falls in this masked border region, flag it in the catalog.
        '''
        off_detector = self.coadd.mask
        npix = self.near_border_kernelsize
        dilation_kernel = np.ones((npix,npix),dtype='bool')
        near_border = binary_dilation(off_detector,dilation_kernel)
        self.photcat_coadd['near_border'] = np.zeros(self.nsources, dtype=bool)
        for i in range(self.nsources):
            s = self.photcat_coadd[i]
            x = np.int32(s['xcentroid'])
            y = np.int32(s['ycentroid'])
            self.photcat_coadd['near_border'][i] = near_border[y,x]
        for b in bands:
            off_detector = self.images[b].mask
            npix = self.near_border_kernelsize
            dilation_kernel = np.ones((npix,npix),dtype='bool')
            near_border = binary_dilation(off_detector,dilation_kernel)
            self.photcats[b]['near_border'] = np.zeros(self.nsources, dtype=bool)
            for i in range(self.nsources):
                s = self.photcats[b][i]
                cs = self.photcat_coadd[i] # Use position on the coadd
                x = np.int32(cs['xcentroid'])
                y = np.int32(cs['ycentroid'])
                self.photcats[b]['near_border'][i] = near_border[y,x]

    def flag_isolated(self):
        '''Extend the bounding box for each source by a small a buffer of a few pixels. 
           The source is considered isolated if there are no other IDs in the segmap within 
           this extended bounding box.
        '''
        self.photcat_coadd['isolated'] = np.ones(self.nsources, dtype=bool)
        buffer = self.flag_isolated_buffer # Grow the bounding box by this amount
        for i,label in enumerate(self.photcat_coadd['label']):
            idx = np.nonzero(self.photometry_coadd.labels == label)[0][0]
            p = self.photometry_coadd[idx]
            yslice = slice(p.bbox_ymin-buffer,p.bbox_ymax+buffer)
            xslice = slice(p.bbox_xmin-buffer,p.bbox_xmax+buffer)
            cutout = self.segm_deblend.data[yslice,xslice]
            ids = cutout.flatten()
            ids = np.array(list(set(ids)),dtype=np.int32)
            ids = ids[ids != 0] # remove the zero
            ids = ids[ids != label] # Remove the ID of this source
            if len(ids) > 0: # Not isolated if other IDs remain in this cutout
                self.photcat_coadd['isolated'][i] = False
