#!/bin/bash
# =============================================================================
# DTI + MSMT-CSD pipeline – truly incremental & resumable
# Skips ANY step whose output already exists
# Includes computation of Directionally Encoded Color (DEC) image using fod2dec
# Fixed compatibility issues for recent MRtrix3 versions
# =============================================================================

#base_dir="/mnt/siat118_disk2/alou/Organized_CN_by_subject"
base_dir="./"

for subject_dir in "$base_dir"/*; do
    [[ -d "$subject_dir" ]] || continue
    subject=$(basename "$subject_dir")
    echo "==================================================================="
    echo "Processing subject: $subject"
    echo "==================================================================="
    OUT_DIR="$subject_dir"

    # ------------------------------------------------------------------
    # Required inputs – if missing, skip entire subject
    # ------------------------------------------------------------------
    # DWI="$subject_dir/data.nii.gz"
    # BVALS="$subject_dir/bvals"
    # BVECS="$subject_dir/bvecs"
    # MASK="$subject_dir/nodif_brain_mask.nii.gz"

    DWI="$subject_dir/DWI_image.nii.gz"
    BVALS="$subject_dir/DWI.bval"
    BVECS="$subject_dir/DWI.bvec"
    MASK="$subject_dir/Mask_image.nii.gz"

    for f in "$DWI" "$BVALS" "$BVECS" "$MASK"; do
        [[ -f "$f" ]] || { echo "ERROR: Missing $f → skipping subject"; continue 2; }
    done

    # ==================================================================
    # 1. Convert to MRtrix format – only if not already done
    # ==================================================================
    if [[ ! -f "$OUT_DIR/dwi.mif" || ! -f "$OUT_DIR/mask.mif" ]]; then
        echo "Converting to .mif format..."
        mrconvert "$DWI" "$OUT_DIR/dwi.mif" -fslgrad "$BVECS" "$BVALS" -force
        mrconvert "$MASK" "$OUT_DIR/mask.mif" -force
    else
        echo "dwi.mif and mask.mif already exist → skipping conversion"
    fi


    if [[ ! -f "$OUT_DIR/FA.nii.gz" || ! -f "$OUT_DIR/MD.nii.gz" || \
          ! -f "$OUT_DIR/AD.nii.gz" || ! -f "$OUT_DIR/RD.nii.gz" ]]; then
        echo "Computing DTI metrics (FA, MD, AD, RD)..."
        dwi2tensor "$OUT_DIR/dwi.mif" "$OUT_DIR/tensor.mif" -mask "$OUT_DIR/mask.mif" -force
        tensor2metric "$OUT_DIR/tensor.mif" \
            -fa  "$OUT_DIR/FA.nii.gz" \
            -rd  "$OUT_DIR/RD.nii.gz" \
            -ad  "$OUT_DIR/AD.nii.gz" \
            -adc "$OUT_DIR/MD.nii.gz" \
            -mask "$OUT_DIR/mask.mif" -force
        rm -f "$OUT_DIR/tensor.mif"  # safe to delete now
    else
        echo "All DTI metrics (FA/MD/AD/RD) already exist → skipping"
    fi

    # ==================================================================
    # 2. Response functions (Dhollander) – only if WM response missing
    # ==================================================================
    # if [[ ! -f "$OUT_DIR/response_wm.txt" ]]; then
    #     echo "Estimating response functions (Dhollander)..."
    #     dwi2response dhollander "$OUT_DIR/dwi.mif" \
    #         "$OUT_DIR/response_wm.txt" \
    #         "$OUT_DIR/response_gm.txt" \
    #         "$OUT_DIR/response_csf.txt" \
    #         -mask "$OUT_DIR/mask.mif" -force
    # else
    #     echo "Response functions already exist → skipping"
    # fi

    # # ==================================================================
    # # 3. MSMT-CSD – only if WM FOD is missing
    # # ==================================================================
    # if [[ ! -f "$OUT_DIR/fod_wm.mif" ]]; then
    #     echo "Running MSMT-CSD..."
    #     dwi2fod msmt_csd "$OUT_DIR/dwi.mif" \
    #         -mask "$OUT_DIR/mask.mif" \
    #         "$OUT_DIR/response_wm.txt" "$OUT_DIR/fod_wm.mif" \
    #         "$OUT_DIR/response_gm.txt" "$OUT_DIR/fod_gm.mif" \
    #         "$OUT_DIR/response_csf.txt" "$OUT_DIR/fod_csf.mif" \
    #         -force
    # else
    #     echo "fod_wm.mif already exists → skipping MSMT-CSD"
    # fi

    # # ==================================================================
    # # 4. Export WM FOD to NIfTI (optional, for external viewing)
    # # ==================================================================
    # if [[ ! -f "$OUT_DIR/fod_wm.nii.gz" ]]; then
    #     echo "Exporting WM FOD to NIfTI..."
    #     mrconvert "$OUT_DIR/fod_wm.mif" "$OUT_DIR/fod_wm.nii.gz" -force
    # fi

    # # ==================================================================
    # # 5. Peak extraction – only if main outputs are missing
    # #    (Updated to use current sh2peaks options: -num_peaks instead of -number)
    # # ==================================================================
    # if [[ ! -f "$OUT_DIR/peaks.nii.gz" ]]; then
    #     echo "Extracting up to 9 FOD peaks..."
    #     sh2peaks "$OUT_DIR/fod_wm.mif" "$OUT_DIR/peaks.nii.gz" \
    #              -mask "$OUT_DIR/mask.mif" -num 9 -threshold 0.10 -force
    # fi

    # if [[ ! -f "$OUT_DIR/peaks5.nii.gz" ]]; then
    #     echo "Extracting up to 5 FOD peaks..."
    #     sh2peaks "$OUT_DIR/fod_wm.mif" "$OUT_DIR/peaks5.nii.gz" \
    #              -mask "$OUT_DIR/mask.mif" -num 5 -threshold 0.07 -force
    # fi

    # if [[ ! -f "$OUT_DIR/peak_amp.nii.gz" ]]; then
    #     echo "Extracting primary peak amplitude..."
    #     sh2peaks "$OUT_DIR/fod_wm.mif" "$OUT_DIR/peak_amp.nii.gz" \
    #              -mask "$OUT_DIR/mask.mif" -num 1 -force
    # fi

    # # ==================================================================
    # # 6. Compute Directionally Encoded Color (DEC) image
    # #    Using dedicated fod2dec command (recommended, modern FOD-based DEC)
    # #    Standard RGB: Red = L-R, Green = A-P, Blue = S-I
    # #    Modulated by FOD integral (default) or primary peak amplitude
    # # ==================================================================
    # if [[ ! -f "$OUT_DIR/dec.nii.gz" ]]; then
    #     echo "Computing DEC image (FOD-based, using fod2dec)..."
    #     fod2dec "$OUT_DIR/fod_wm.mif" "$OUT_DIR/dec.nii.gz" -mask "$OUT_DIR/mask.mif" -force
    #     # Alternative: modulate by primary peak amplitude instead of FOD integral
    #     # fod2dec "$OUT_DIR/fod_wm.mif" "$OUT_DIR/dec.nii.gz" -mask "$OUT_DIR/mask.mif" -contrast "$OUT_DIR/peak_amp.nii.gz" -force
    # else
    #     echo "DEC image (dec.nii.gz) already exists → skipping"
    # fi

    # # ==================================================================
    # # Cleanup intermediate files (optional, comment out if you want to keep them)
    # # ==================================================================
    # rm -f "$OUT_DIR"/response_*.txt
    # rm -f "$OUT_DIR"/fod_gm.mif
    # rm -f "$OUT_DIR"/fod_csf.mif
    rm -f "$OUT_DIR"/dwi.mif  # Uncomment to save space

    echo "FINISHED $subject – all missing steps completed!"
    echo "==================================================================="
done

echo "All subjects processed (incrementally)!"
