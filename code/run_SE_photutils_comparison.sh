datalad run -m "SE_photutils_comparison" \
    --input 'outputs/dr0.5_c0.1/*.ecsv' \
    --output 'outputs/dr0.5_c0.1/*_kron_color_comparison.pdf' \
    "python code/dr05_c01_SE_photutils_comparison.py"

