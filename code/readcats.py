# Reading in and merging CEERS catalogs
from os import path
from astropy.table import Table, vstack
from dataclasses import dataclass
from astropy.io import fits
import bz2
import dill
import numpy as np
from photutils import SourceCatalog, SegmentationImage

def load_segm(data,filename):
    with bz2.BZ2File(filename, 'rb') as f:
       dill.load(data, f)

@dataclass
class CEERSCat:
    version: str = 'dr0.5_c0.1'
    catdir: str = 'outputs/dr0.5_c0.1'
    fields: list = (1,2,3,6)
    # Skipping f105w because it doesn't exist for all fields
    bands:  list = ('f606w','f814w','f125w','f140w','f160w',
                    'f115w','f150w','f200w','f277w','f356w','f410m','f444w')
    refband: str = 'f277w'
    ab_zpt: float = 31.4 # 1 nJy in ABMag


    def __post_init__(self):
        self.phot = {} # Merged for all fields
        self.mphot = {} # Merged for all fields
        self.seg = {} # Separate for each field
        self.photom = {} # Separate for each field
        self.seg_image = {} # Separate for each field
        self.matchbands = {
                'f115w': 'f115w_f277w',
                'f150w': 'f150w_f277w',
                'f200w': 'f200w_f277w',
                'f356w': 'f277w_f356w',
                'f410m': 'f277w_f410m',
                'f444w': 'f277w_f444w',
                'f606w': 'f606w_f277w',
                'f814w': 'f814w_f277w',
                'f125w': 'f277w_f125w',
                'f140w': 'f277w_f140w',
                'f160w': 'f277w_f160w',
        }

    def read_detcats(self):
        detcat = None
        for f in self.fields:
            filename = path.join(self.catdir,f"nircam{f}_detect_f277w_f356w__{self.version}.ecsv")
            cat = Table.read(filename)
            cat['field'] = np.full(len(cat),f)
            if detcat is None:
                detcat = cat
            else:
                detcat = vstack((detcat,cat))
        self.det = detcat

    def read_photcat_band(self,band):
        photcat = None
        for f in self.fields:
            filename = path.join(self.catdir,f"nircam{f}_{band}_{self.version}.ecsv")
            cat = Table.read(filename)
            cat['field'] = np.full(len(cat),f)
            if photcat is None:
                photcat = cat
            else:
                photcat = vstack((photcat,cat))
        self.phot[band] = photcat
        self.add_kron_mags(self.phot[band])

    def read_matched_photcat_band(self,band):
        photcat = None
        for f in self.fields:
            filts = self.matchbands[band]
            filename = path.join(self.catdir,f"nircam{f}_matched_{filts}_{self.version}.ecsv")
            cat = Table.read(filename)
            cat['field'] = np.full(len(cat),f)
            if photcat is None:
                photcat = cat
            else:
                photcat = vstack((photcat,cat))
        self.mphot[band] = photcat


    def read_photcats(self):
        for b in self.bands:
            self.read_photcat_band(b)

    def read_matched_photcats(self):
        for b in self.bands:
            if b != self.refband: # No matched cat for this one
                self.read_matched_photcat_band(b)

    def correct_to_reference_band(self):
        rphot = self.phot[self.refband] # Reference band catalog
        self.mphot[self.refband] = self.phot[self.refband]
        for b in self.matchbands:
            mphot = self.mphot[b]
            phot = self.phot[b]
            b1,b2 = self.matchbands[b].split('_')
            # If the first band is the reference band
            # Multiply flux by the ratio of unconvolved/convolved reference band
            # Otherwise, matched photometry is already matched to the reference band
            if (b1 == self.refband):
                for column in phot.colnames:
                    if column[-5:] == "_flux":
                        mphot[column] = phot[column] * (rphot[column]/mphot[column])
            self.add_kron_mags(mphot)

    def add_kron_mags(self,cat):
        for column in cat.colnames:
            if column[-5:] == "_flux":
                fluxname = column[:-5]
                cat[f"{fluxname}_mag"] = -2.5*np.log10(column) + self.ab_zpt

    def read_all_cats(self):
        self.read_detcats()
        self.read_photcats()
        self.read_matched_photcats()
        self.correct_to_reference_band()

    def read_seg_images(self):
        for f in self.fields:
            filename = path.join(self.catdir,f"nircam{f}_seg_{self.version}.fits")
            self.seg_image[f] = fits.getdata(filename)

    def read_segs(self):
        for f in self.fields:
            filename = path.join(self.catdir,f"nircam{f}_seg_object_{self.version}.pbz2")
            with bz2.BZ2File(filename, 'rb') as fp:
                self.seg[f] = dill.load(fp)

    def read_photoms(self):
        for f in self.fields:
            filename = path.join(self.catdir,f"nircam{f}_photom_object_coadd_{self.version}.pbz2")
            with bz2.BZ2File(filename, 'rb') as fp:
                self.photom[f] = dill.load(fp)


