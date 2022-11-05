All custom code goes into this directory. All scripts should be written such
that they can be executed from the root of the dataset, and are only using
relative paths for portability.

To read the catalogs from this directory:

```
    import readcats
    c = readcats.CEERSCat(catdir="../outputs/dr0.5_c0.1")
    c.read_all_cats()
```

Scripts are as follows:
  - readcats.py: read in the catalogs
  - `dr05_c01.py`:  `python dr05_c01.py field_number` to run on a given field
  - `dr05paths.py`: has all of the pathnames needed for making the catalog
  - `make_catalog.py`: The script that does all the work (called from dr05_c01.py)
  - `get_detmaps.py`: download detector maps from google drive
  - `get_psf_kernels.py`: download psf kernels from google drive
