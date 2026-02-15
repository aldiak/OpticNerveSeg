# OpticNerveSeg: Public Dataset for Cranial Nerve II Segmentation from Multimodal MRI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

## 📋 Overview

**OpticNerveSeg** is the first large-scale, publicly available dataset for cranial nerve II (optic nerve) segmentation from multimodal MRI. The dataset comprises 151 subjects from the Human Connectome Project (HCP) Young Adult cohort, including high-resolution T1-weighted structural MRI, multi-shell diffusion MRI, and high-quality binary segmentation masks of the bilateral cranial nerve II, optic chiasm, and optic tracts.

This repository contains all code for the annotation pipeline, preprocessing scripts, and baseline semi-supervised segmentation methods described in our paper:

> Diakite, A., Li, C., & Wang, S. (2025). OpticNerveSeg: A Public Dataset for Automatic Segmentation of the Cranial Nerve II from Multimodal MRI. *Scientific Data* (under review).

## 🎯 Key Features

- **151 subjects** with high-quality binary masks
- **Multimodal data**: T1w structural MRI + multi-shell diffusion MRI (b=1000,2000,3000 s/mm²)
- **Anatomical coverage**: cranial nerve II including optic nerve, optic chiasm, and optic tracts
- **Rich metadata**: mean radius, curvature, length, SNR, contrast, partial volume indices
- **Human-in-the-loop annotations**: 16 expert tractography seeds + semi-supervised inference + expert correction
- **76% reduction** in expert annotation time compared to full manual segmentation
- **Semi-supervised baselines** with 5% and 20% labeled data for 8 state-of-the-art methods

## 📊 Dataset Access

The dataset is available on Zenodo: [https://zenodo.org/records/XXXXXXX](https://zenodo.org/records/XXXXXXX)

**Contents:**
- 151 binary mask files (`.nii.gz`) at 1. mm isotropic resolution
- Metadata CSV with subject-level anatomical and image quality features

**Note:** Raw HCP images are not redistributed due to data use terms. Users must obtain the original data from [ConnectomeDB](https://db.humanconnectome.org).

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/siat-nlp/OpticNerveSeg.git
cd OpticNerveSeg
pip install -r requirements.txt
```
## 📁 Repository Structure
OpticNerveSeg/
├── masks/                     # Binary segmentation masks (151 subjects)
│   ├── 100206_label.nii.gz
│   ├── 100307_label.nii.gz
│   └── ...
├── metadata/                  # Subject-level metadata
│   └── subjects_metadata.csv
├── scripts/                   # All code
│   ├── tractography/          # MRtrix3 tractography pipeline
│   │   ├── 01_estimate_fod.sh
│   │   ├── 02_tractography.sh
│   │   └── 03_voxelize.sh
│   ├── semi_supervised/       # Semi-supervised baseline methods
│   │   ├── dataloader/
│   │   ├── utils/
│   │   ├── networks/
│   │   ├── test_utils.py
│   │   ├── train.py
│   │   ├── infer.py
│   │   └── run_all_alg.py
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
└── README.md                  # This file
