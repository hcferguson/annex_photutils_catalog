#!/usr/bin/env python

# Compare SExtractor and Photutils catalogs

__author__ = "Henry C. Ferguson, STScI"
__version__ = "0.0.1"
__license__ = "BSD3"

# Version history
# 0.1 Adapted from software/jwst_notebooks/SE_photutils_v0.2_c0.2_comparison.ipynb

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
from astropy.io import fits
from astropy.stats import biweight_location, biweight_scale
import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.signal import medfilt

import sys
from os import path

# Import catalog reading script
sys.path.append('/Volumes/CEERS2/ceersdl/photutils_catalog/code')
import readcats

def seband(band):
    return band[:-1].upper()

def compare_colors(rc,secat,photutils_subset,se_subset,band0,band1='f277w'):
    ''' Compare photutils and SE colors '''
    phcolor = rc.mphot[band0]['kron_mag'][photutils_subset] - rc.mphot[band1]['kron_mag'][photutils_subset]
    secolor = ab(secat[seband(band0)][se_subset]) - ab(secat[seband(band1)][se_subset])
    return phcolor-secolor

def binned_stats(x,y,xmin,xmax,xstep):
    ''' Bins along the x axis and estimates for y the robust mean, std, stderr and IQR '''
    xvals = np.arange(xmin+xstep/2.,xmax,xstep)
    yvals = []
    ystderr = []
    ystd = []
    iqr0 = []
    iqr1 = []
    for xbin in xvals:
        sample = (x > xbin-xstep/2.) & (x <= xbin + xstep/2.)
        sample = sample & ~np.isnan(y)
        # print(xbin,sample.sum())
        yvals += [biweight_location(y[sample])]
        ystderr += [biweight_scale(y[sample])/(len(y[sample]) -1)]
        ystd += [biweight_scale(y[sample])]
        iqr0 += [np.percentile(np.copy(y[sample]),25)] # throws an error if I don't copy the array
        iqr1 += [np.percentile(np.copy(y[sample]),75)]
    yvals = np.array(yvals)
    ystderr = np.array(ystderr)
    ystd = np.array(ystd)
    iqr0,iqr1 = np.array(iqr0), np.array(iqr1)
    return xvals,yvals,ystderr,ystd,iqr0,iqr1

def plot_compare_colors(rc,secat,photutils_subset,se_subset,bands):
    ''' Plot a comparison of the colors '''
    plt.figure(figsize=(15,15))
    plt.title('SE - photutils colors')
    for i,b in enumerate(bands):
        plt.subplot(2,3,i+1)
        diff = compare_colors(rc,secat,photutils_subset,se_subset,b)
        mag = ab(secat['F277'][se_subset])
        plt.scatter(mag,diff,alpha=0.1)
        plt.xlim(18,29)
        plt.ylim(-0.65,0.65)
        binned_mag, binned_diff, binned_stderr, binned_std, iqr0, iqr1 = binned_stats(mag,diff,18,29,0.5)
        plt.errorbar(binned_mag,binned_diff,np.array([binned_diff-iqr0,iqr1-binned_diff]),alpha=0.7,color='r')
        plt.grid()
        plt.xlabel('f277w mag')
        plt.text(19,0.55,f"{b} - f277w color",fontsize=14)


if __name__ == "__main__":

    # Bands and fields
    bands = ['f115w','f150w','f200w','f277w','f356w','f410m','f444w']
    fields = [1,2,3,6]
    # Read the SExtractor catalog
    secatdir = '/Volumes/CEERS2/ceers/CEERSJune2022/v0.2/SteveF_cats'
    outdir = "outputs/dr0.5_c0.1"
    _cats = []
    for f in fields:
        catfile = f"CEERS_NIRCam{f}_v0.2_photom.fits"
        _cat = Table.read(path.join(secatdir,catfile))
        _cat['field'] = np.full(len(_cat),f,dtype=np.int32)
        _cats += [_cat]
    secat = vstack(_cats)
    # Add a coordinates column
    secat['coords'] = SkyCoord(secat['RA'],secat['DEC'],unit=['degree','degree'],frame='icrs')
    
    # Read the photutils catalogs
    rc = readcats.CEERSCat()
    rc.read_all_cats()
    
    # Cross-match the catalogs
    idx, d2d, d3d = secat['coords'].match_to_catalog_sky(rc.det['sky_centroid'])
    select = d2d < 0.1 * u.arcsecond
    se_subset = select
    photutils_subset = idx[select]
    assert len(photutils_subset) == len(secat[se_subset])
    print(f"Matched photutils,SE: {len(photutils_subset)}")

    # NIRCam colors 
    plot_compare_colors(rc,secat,photutils_subset,se_subset,['f115w','f150w','f200w','f356w','f410m','f444w'])
    plt.savefig(path.join(outdir,"SE_photutils_dr0.5_c0.1_nircam_kron_color_comparison.pdf"))

    # HST colors 
    plot_compare_colors(rc,secat,photutils_subset,se_subset,['f606w','f814w','f105w','f125w','f160w'])
    plt.savefig(path.join(outdir,"SE_photutils_dr0.5_c0.1_hst_kron_color_comparison.pdf"))
