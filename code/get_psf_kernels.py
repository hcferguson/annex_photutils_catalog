

import datalad.api as dl
import os

psf_kernel_links = {
 "kernel_f814w_f277w.fits": "https://drive.google.com/file/d/19W2uNQH5SFWLT_GpNidV7wc9X3TmF0mF/view?usp=share_link",
 "kernel_f277w_f410m.fits": "https://drive.google.com/file/d/1jR3SFDObEHuUpzn61KwU_5qj3LVPseyG/view?usp=share_link",
 "kernel_f277w_f356w.fits": "https://drive.google.com/file/d/1G1Iev4MK7PjBt0OlwzllWQ_XTsc1Ks0_/view?usp=share_link",
 "kernel_f606w_f277w.fits": "https://drive.google.com/file/d/15rNAOdzArteDjcMS7zP-SQcKPtiWiPV_/view?usp=share_link",
 "kernel_f277w_f444w.fits": "https://drive.google.com/file/d/11WDkgUEVjoe6H-lKXEXOATMYSh5K29cl/view?usp=share_link",
 "kernel_f277w_f125w.fits": "https://drive.google.com/file/d/1oH2IasqdDXaGJX75EFUFUFdLd2_1OwZw/view?usp=share_link",
 "kernel_f277w_f140w.fits": "https://drive.google.com/file/d/1VdYwt5ctn5_fomBY6CvSWkjS4SUG1Slb/view?usp=share_link",
 "kernel_f115w_f277w.fits": "https://drive.google.com/file/d/1PYPxo_0XdWuy_4zwsanydDxQU092L5G3/view?usp=share_link",
 "kernel_f150w_f277w.fits": "https://drive.google.com/file/d/1N9EUf09kbOCh3c40EVS6Qc9dFZORjDPN/view?usp=share_link",
 "kernel_f277w_f160w.fits": "https://drive.google.com/file/d/15cSRz6_j5W7Z124CXQBTlJ7eUn0aBK-i/view?usp=share_link",
 "kernel_f277w_f105w.fits": "https://drive.google.com/file/d/10g8J8dBrLYIsqmU3DrayYx7QG-GiU3lW/view?usp=share_link",
 "kernel_f200w_f277w.fits": "https://drive.google.com/file/d/1O_XMvwHjZgVQGjHNsjia1rH2xqoZ3ChP/view?usp=share_link"
 }
destination_dir = "inputs/psfs"

if __name__ == "__main__":
    if not os.path.exists(destination_dir):
        os.mkdir(destination_dir)
    for k in psf_kernel_links:
        url = psf_kernel_links[k]
        destination = os.path.join(destination_dir,k)
        print(k)
        dl.download_url(url,message=f"Downloaded {k}",path=destination)


