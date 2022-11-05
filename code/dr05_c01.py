#!/usr/bin/env python
# coding: utf-8


__author__ = "Henry C. Ferguson, STScI"
__version__ = "0.1.0"
__license__ = "BSD3"

# Version notes
# 0.1.0 -- Copied and modified from software/jwst_scripts/CEERS_catalogs/v0.2_c0.2.py
# Intended to find faint small objects.

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits

import sys
from os import path
from pprint import pprint


############### Customize for each new run on a different dataset ##############
# Scripts that might need to be customized for different data versions
import dr05paths as paths
from make_catalog import PhotCat

# Input and output directory and filename suffix 
indir = "inputs"
outdir= "outputs/dr0.5_c0.1"
suffix = f"_dr0.5_c0.1"
################################################################################


# Set up the catalog object 
ppc = PhotCat()

# If you wish to change any of these, set them and then
# re-execute pc.populate_catalog_metadata *before* reading the images
# Otherwise all the pixels will be in the catalog metadata
pprint(ppc.catalog_metadata)

if __name__ == "__main__":
    field = sys.argv[1]

    # Read the files for field 1
    # Set up the catalog object separately for each field
    pc = PhotCat()
    # Read the JWST data
    pc.read_jwst(paths.ceerspaths,field)
    # Read the HST data
    pc.read_hst(paths.ceerspaths,field)

    # Make the coadd
    pc.make_coadd()
    # Run the detection
    pc.detect()
    # Do the photometry on the convolved detection image
    pc.photometry_on_convolved_detect_image()
    # Add extra output columns
    pc.add_extra_columns()
    # Do the photometry on the individual bands 
    pc.photometry_bands(pc.jwst_bands)
    pc.photometry_bands(pc.hst_bands)
    # Photometry on PSF-matched images
    pc.create_matched(paths.psf_kernels)
    pc.photometry_matched_bands()
    # Large Kron-aperture photometry
    pc.extra_kron_photometry(pc.jwst_bands)
    pc.extra_kron_photometry(pc.hst_bands)
    # Large Kron-aperture photometry for the matched images
    pc.extra_kron_matched_photometry( list(pc.matched_images.keys()) )
    # Fraction of light radii
    pc.fluxfrac_radii([0.2,0.5,0.8],pc.jwst_bands)
    pc.fluxfrac_radii([0.2,0.5,0.8],pc.hst_bands)

    # Flag objects near an image border
    pc.flag_near_border(pc.jwst_bands)
    pc.flag_near_border(pc.hst_bands)
    # Identify the detector for each source
    pc.identify_detectors()

    # Write out the catalogs, coadd image, segmap and segmentation and 
    # photometry data structures
    prefix = f"nircam{field}_"
    pc.write_coadd(outputdir=outdir,prefix=prefix,suffix=suffix)
    pc.write_segmentation_image(outputdir=outdir,prefix=prefix,suffix=suffix)
    pc.write_coadd_photcat(outputdir=outdir,prefix=prefix,suffix=suffix)
    pc.write_photcat(outputdir=outdir,prefix=prefix,suffix=suffix)
    pc.write_matched_photcat(outputdir=outdir,prefix=prefix,suffix=suffix)
    pc.dill_coadd_photometry(outputdir=outdir,prefix=prefix,suffix=suffix)
    pc.dill_segmap(outputdir=outdir,prefix=prefix,suffix=suffix)

    # Don't know why I need to do this, but on the second iteration pc still has the attribute
    # Maybe the global overrode the local?
    del pc.my_photcolumns
