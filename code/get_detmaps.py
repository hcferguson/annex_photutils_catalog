import datalad.api as dl
import os

detmap_links  = {
 "ceers_nircam3_detmap.fits.gz": "https://drive.google.com/file/d/1z_eZl9rrZyJFvBxemgkKuLzcuL2a4dEi/view?usp=share_link",
 "ceers_nircam6_detmap.fits.gz": "https://drive.google.com/file/d/1x_QVuVojeIm0aKIhU_yfUcllu0-dAWWC/view?usp=share_link", 
 "ceers_nircam2_detmap.fits.gz": "https://drive.google.com/file/d/1uEjutDHw0lb0QYBcsvZdhfNpMKNzy4Sf/view?usp=share_link",
 "ceers_nircam1_detmap.fits.gz": "https://drive.google.com/file/d/11uS87hh8mpipwZQJSDxqY8LbGL-rsFC2/view?usp=share_link" 
 }
destination_dir = "inputs/detector_maps"

if __name__ == "__main__":
    if not os.path.exists(destination_dir):
        os.mkdir(destination_dir)
    for k in detmap_links:
        url = detmap_links[k]
        destination = os.path.join(destination_dir,k)
        print(k)
        dl.download_url(url,message=f"Downloaded {k}",path=destination)
