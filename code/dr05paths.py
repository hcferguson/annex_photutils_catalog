# All the configurable stuff for dr0.5
from os import path
from dataclasses import dataclass, asdict
from astropy.io import fits
from astropy.wcs import WCS

#################### Customize for each data set ####################
jwst_datadir = './inputs/dr0.5'
hst_30mas_dir = './inputs/dr0.5'
detmap_datadir = './inputs/detector_maps'
kerneldir = './inputs/psfs'

nircam1 = {
    'fitsfile': {
             'f115w': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam1_f115w_dr0.5_i2d.fits.gz'),
             'f200w': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam1_f200w_dr0.5_i2d.fits.gz'),
             'f150w': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam1_f150w_dr0.5_i2d.fits.gz'),
             'f277w': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam1_f277w_dr0.5_i2d.fits.gz'),
             'f356w': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam1_f356w_dr0.5_i2d.fits.gz'),
             'f410m': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam1_f410m_dr0.5_i2d.fits.gz'),
             'f444w': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam1_f444w_dr0.5_i2d.fits.gz'),
             'f606w': path.join(hst_30mas_dir,'egs_all_acs_wfc_f606w_030mas_v1.9_nircam1_mef.fits.gz'),
             'f814w': path.join(hst_30mas_dir,'egs_all_acs_wfc_f814w_030mas_v1.9_nircam1_mef.fits.gz'),
             'f125w': path.join(hst_30mas_dir,'egs_all_wfc3_ir_f125w_030mas_v1.9_nircam1_mef.fits.gz'),
             'f140w': path.join(hst_30mas_dir,'egs_all_wfc3_ir_f140w_030mas_v1.9_nircam1_mef.fits.gz'),
             'f160w': path.join(hst_30mas_dir,'egs_all_wfc3_ir_f160w_030mas_v1.9_nircam1_mef.fits.gz'),
            },
    'bksub_ext': {
             'f115w': 1,
             'f150w': 1,
             'f200w': 1,
             'f277w': 1,
             'f356w': 1,
             'f410m': 1,
             'f444w': 1,
             'f606w': 1,
             'f814w': 1,
             'f125w': 1,
             'f140w': 1,
             'f160w': 1,
            },
    'rms_ext': {
             'f115w': 3,
             'f150w': 3,
             'f200w': 3,
             'f277w': 3,
             'f356w': 3,
             'f410m': 3,
             'f444w': 3,
             'f606w': 3,
             'f814w': 3,
             'f125w': 3,
             'f140w': 3,
             'f160w': 3
            },
    'tiermask_ext': {
             'f115w': 10,
             'f150w': 10,
             'f200w': 10,
             'f277w': 10,
             'f356w': 10,
             'f410m': 10,
             'f444w': 10,
             'f606w': 5,
             'f814w': 5,
             'f125w': 5,
             'f140w': 5,
             'f160w': 5
    'detmap': path.join(detmap_datadir,'ceers_nircam1_detmap.fits.gz')
    }

nircam2 = {
    'fitsfile': {
             'f115w': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam2_f115w_dr0.5_i2d.fits.gz'),
             'f200w': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam2_f200w_dr0.5_i2d.fits.gz'),
             'f150w': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam2_f150w_dr0.5_i2d.fits.gz'),
             'f277w': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam2_f277w_dr0.5_i2d.fits.gz'),
             'f356w': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam2_f356w_dr0.5_i2d.fits.gz'),
             'f410m': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam2_f410m_dr0.5_i2d.fits.gz'),
             'f444w': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam2_f444w_dr0.5_i2d.fits.gz'),
             'f606w': path.join(hst_30mas_dir,'egs_all_acs_wfc_f606w_030mas_v1.9_nircam2_mef.fits.gz'),
             'f814w': path.join(hst_30mas_dir,'egs_all_acs_wfc_f814w_030mas_v1.9_nircam2_mef.fits.gz'),
             'f105w': path.join(hst_30mas_dir,'egs_all_wfc3_ir_f105w_030mas_v1.9_nircam2_mef.fits.gz'),
             'f125w': path.join(hst_30mas_dir,'egs_all_wfc3_ir_f125w_030mas_v1.9_nircam2_mef.fits.gz'),
             'f140w': path.join(hst_30mas_dir,'egs_all_wfc3_ir_f140w_030mas_v1.9_nircam2_mef.fits.gz'),
             'f160w': path.join(hst_30mas_dir,'egs_all_wfc3_ir_f160w_030mas_v1.9_nircam2_mef.fits.gz'),
            },
    'bksub_ext': {
             'f115w': 1,
             'f150w': 1,
             'f200w': 1,
             'f277w': 1,
             'f356w': 1,
             'f410m': 1,
             'f444w': 1,
             'f606w': 1,
             'f814w': 1,
             'f105w': 1,
             'f125w': 1,
             'f140w': 1,
             'f160w': 1,
            },
    'rms_ext': {
             'f115w': 3,
             'f150w': 3,
             'f200w': 3,
             'f277w': 3,
             'f356w': 3,
             'f410m': 3,
             'f444w': 3,
             'f606w': 3,
             'f814w': 3,
             'f105w': 3,
             'f125w': 3,
             'f140w': 3,
             'f160w': 3
            },
    'tiermask_ext': {
             'f115w': 10,
             'f150w': 10,
             'f200w': 10,
             'f277w': 10,
             'f356w': 10,
             'f410m': 10,
             'f444w': 10,
             'f606w': 5,
             'f814w': 5,
             'f105w': 5,
             'f125w': 5,
             'f140w': 5,
             'f160w': 5
    'detmap': path.join(detmap_datadir,'ceers_nircam2_detmap.fits.gz')
    }

nircam3 = {
    'fitsfile': {
             'f115w': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam3_f115w_dr0.5_i2d.fits.gz'),
             'f200w': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam3_f200w_dr0.5_i2d.fits.gz'),
             'f150w': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam3_f150w_dr0.5_i2d.fits.gz'),
             'f277w': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam3_f277w_dr0.5_i2d.fits.gz'),
             'f356w': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam3_f356w_dr0.5_i2d.fits.gz'),
             'f410m': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam3_f410m_dr0.5_i2d.fits.gz'),
             'f444w': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam3_f444w_dr0.5_i2d.fits.gz'),
             'f606w': path.join(hst_30mas_dir,'egs_all_acs_wfc_f606w_030mas_v1.9_nircam3_mef.fits.gz'),
             'f814w': path.join(hst_30mas_dir,'egs_all_acs_wfc_f814w_030mas_v1.9_nircam3_mef.fits.gz'),
             'f105w': path.join(hst_30mas_dir,'egs_all_wfc3_ir_f105w_030mas_v1.9_nircam3_mef.fits.gz'),
             'f125w': path.join(hst_30mas_dir,'egs_all_wfc3_ir_f125w_030mas_v1.9_nircam3_mef.fits.gz'),
             'f140w': path.join(hst_30mas_dir,'egs_all_wfc3_ir_f140w_030mas_v1.9_nircam3_mef.fits.gz'),
             'f160w': path.join(hst_30mas_dir,'egs_all_wfc3_ir_f160w_030mas_v1.9_nircam3_mef.fits.gz'),
            },
    'bksub_ext': {
             'f115w': 1,
             'f150w': 1,
             'f200w': 1,
             'f277w': 1,
             'f356w': 1,
             'f410m': 1,
             'f444w': 1,
             'f606w': 1,
             'f814w': 1,
             'f105w': 1,
             'f125w': 1,
             'f140w': 1,
             'f160w': 1,
            },
    'rms_ext': {
             'f115w': 3,
             'f150w': 3,
             'f200w': 3,
             'f277w': 3,
             'f356w': 3,
             'f410m': 3,
             'f444w': 3,
             'f606w': 3,
             'f814w': 3,
             'f105w': 3,
             'f125w': 3,
             'f140w': 3,
             'f160w': 3
            },
    'tiermask_ext': {
             'f115w': 10,
             'f150w': 10,
             'f200w': 10,
             'f277w': 10,
             'f356w': 10,
             'f410m': 10,
             'f444w': 10,
             'f606w': 5,
             'f814w': 5,
             'f105w': 5,
             'f125w': 5,
             'f140w': 5,
             'f160w': 5
    'detmap': path.join(detmap_datadir,'ceers_nircam3_detmap.fits.gz')
    }


nircam6 = {
    'fitsfile': {
             'f115w': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam6_f115w_dr0.5_i2d.fits.gz'),
             'f200w': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam6_f200w_dr0.5_i2d.fits.gz'),
             'f150w': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam6_f150w_dr0.5_i2d.fits.gz'),
             'f277w': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam6_f277w_dr0.5_i2d.fits.gz'),
             'f356w': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam6_f356w_dr0.5_i2d.fits.gz'),
             'f410m': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam6_f410m_dr0.5_i2d.fits.gz'),
             'f444w': path.join(jwst_datadir,'hlsp_ceers_jwst_nircam_nircam6_f444w_dr0.5_i2d.fits.gz'),
             'f606w': path.join(hst_30mas_dir,'egs_all_acs_wfc_f606w_030mas_v1.9_nircam6_mef.fits.gz'),
             'f814w': path.join(hst_30mas_dir,'egs_all_acs_wfc_f814w_030mas_v1.9_nircam6_mef.fits.gz'),
             'f105w': path.join(hst_30mas_dir,'egs_all_wfc3_ir_f105w_030mas_v1.9_nircam6_mef.fits.gz'),
             'f125w': path.join(hst_30mas_dir,'egs_all_wfc3_ir_f125w_030mas_v1.9_nircam6_mef.fits.gz'),
             'f140w': path.join(hst_30mas_dir,'egs_all_wfc3_ir_f140w_030mas_v1.9_nircam6_mef.fits.gz'),
             'f160w': path.join(hst_30mas_dir,'egs_all_wfc3_ir_f160w_030mas_v1.9_nircam6_mef.fits.gz'),
            },
    'bksub_ext': {
             'f115w': 1,
             'f150w': 1,
             'f200w': 1,
             'f277w': 1,
             'f356w': 1,
             'f410m': 1,
             'f444w': 1,
             'f606w': 1,
             'f814w': 1,
             'f105w': 1,
             'f125w': 1,
             'f140w': 1,
             'f160w': 1,
            },
    'rms_ext': {
             'f115w': 3,
             'f150w': 3,
             'f200w': 3,
             'f277w': 3,
             'f356w': 3,
             'f410m': 3,
             'f444w': 3,
             'f606w': 3,
             'f814w': 3,
             'f105w': 3,
             'f125w': 3,
             'f140w': 3,
             'f160w': 3
            },
    'tiermask_ext': {
             'f115w': 10,
             'f150w': 10,
             'f200w': 10,
             'f277w': 10,
             'f356w': 10,
             'f410m': 10,
             'f444w': 10,
             'f606w': 5,
             'f814w': 5,
             'f105w': 5,
             'f125w': 5,
             'f140w': 5,
             'f160w': 5
    'detmap': path.join(detmap_datadir,'ceers_nircam6_detmap.fits.gz')
    }


ceerspaths = {1:nircam1, 2:nircam2, 3:nircam3, 4:nircam4}

psf_kernels = {
  'f105w': path.join(kernel_dir,'kernel_f277w_f105w.fits'),
  'f115w': path.join(kernel_dir,'kernel_f115w_f277w.fits'),
  'f125w': path.join(kernel_dir,'kernel_f277w_f125w.fits'),
  'f140w': path.join(kernel_dir,'kernel_f277w_f140w.fits'),
  'f150w': path.join(kernel_dir,'kernel_f150w_f277w.fits'),
  'f160w': path.join(kernel_dir,'kernel_f277w_f160w.fits'),
  'f200w': path.join(kernel_dir,'kernel_f200w_f277w.fits'),
  'f356w': path.join(kernel_dir,'kernel_f277w_f356w.fits'),
  'f410m': path.join(kernel_dir,'kernel_f277w_f410m.fits'),
  'f444w': path.join(kernel_dir,'kernel_f277w_f444w.fits'),
  'f606w': path.join(kernel_dir,'kernel_f606w_f277w.fits'),
  'f814w': path.join(kernel_dir,'kernel_f814w_f277w.fits'),
}
############################################################################3

def which_footprint(coords,band='f200w'):
    global ceerspaths
    for f in ceerspaths.keys():
        with fits.open(ceerspaths[f]['sci'][band]) as hdu:
            wcs = WCS(hdu[ceerspaths[f]['sci_ext'][band]].header)
            if wcs.footprint_contains(coords):
                return f

def get_wcs(field,band):
    global ceerspaths
    with fits.open(ceerspaths[field]['sci'][band]) as hdu:
        wcs = WCS(hdu[ceerspaths[field]['sci_ext'][band]].header)
    return wcs

def get_data(field,band):
    global ceerspaths
    with fits.open(ceerspaths[field]['sci'][band]) as hdu:
        sci = hdu[ceerspaths[field]['sci_ext'][band]].data
        wcs = WCS(hdu[ceerspaths[field]['sci_ext'][band]].header)
    rms = fits.getdata(ceerspaths[field]['rms'][band],ceerspaths[field]['rms_ext'][band])
    return sci,rms,wcs

def get_tiermask(field,band):
    global ceerspaths
    with fits.open(ceerspaths[field]['sci'][band]) as hdu:
        mask = hdu[ceerspaths[field]['tiermask_ext'][band]].data
    return mask

def get_edgemask(field,band):
    global ceerspaths
    with fits.open(ceerspaths[field]['sci'][band]) as hdu:
        mask = hdu[ceerspaths[field]['tiermask_ext'][band]].data
    mask = bitwise_and(mask,1) == 1
    return mask
