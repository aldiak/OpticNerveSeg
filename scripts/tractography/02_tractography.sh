#!/bin/bash

# Set the base directory where the dataset is located
base_dir="/home/alou/Desktop/hcp_new_1/103818"
# Set the output directory for the registered images
output_dir="/home/alou/Desktop/hcp_new_1/103818"
# Set the base directory where the ROI is located
#roi_dir = "/home/alou/Desktop/hcp_new_1"
roi_dir="/home/alou/Desktop/hcp_new_1/103818"
T1_image="$output_dir/T1w_acpc_dc_restore_1.25.nii.gz"
ROI_T1_out1="$output_dir/ROI_T1_MNI1.nii.gz"
DWI_image="$output_dir/Diffusion/data.nii.gz"
ROI_image1="$roi_dir/ROI.nii.gz"
ROI_image2="$roi_dir/ROA.nii.gz"


#mrconvert $T1_image $output_dir/T1.mif -force
##
#mrconvert $DWI_image $output_dir/DWI.mif -fslgrad $output_dir/Diffusion/bvecs $output_dir/Diffusion/bvals -force
##
#dwi2response tournier $output_dir/DWI.mif $output_dir/response.txt -force
##
#dwi2fod csd $output_dir/DWI.mif $output_dir/response.txt $output_dir/wm_fod.mif -force

5ttgen fsl $output_dir/T1.mif $output_dir/5tt.mif -force
dwi2response msmt_5tt $output_dir/DWI.mif $output_dir/5tt.mif $output_dir/wm.txt $output_dir/gm.txt $output_dir/csf.txt -force
dwi2fod msmt_csd $output_dir/DWI.mif $output_dir/wm.txt $output_dir/wm_fod.mif $output_dir/gm.txt $output_dir/gm_fod.mif $output_dir/csf.txt $output_dir/csf_fod.mif -force

tckgen -algorithm iFOD1 -select 1000000 -angle 10 -maxlen 1000 -step 0.5 -seed_image $ROI_T1_out1 -nthreads 30  $output_dir/wm_fod.mif -seed_cutoff 0.006 $output_dir/Fiber_ifod1_ocn.tck -cutoff 0.006 -force

tckedit $output_dir/Fiber_ifod1_ocn.tck $output_dir/Fiber_ifod1_ocn.tck -include $output_dir/ROI.mif -include $output_dir/ROA.mif -force
