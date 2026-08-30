# OpticNerveSeg: Public Dataset for Cranial Nerve II Segmentation from Multimodal MRI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: BALSA](https://img.shields.io/badge/Dataset-BALSA-blue.svg)](https://balsa.wustl.edu/study/N9G6D)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-ee4c2c.svg)](https://pytorch.org/)

## 📋 Overview

**OpticNerveSeg** is the first large-scale, publicly available dataset for cranial nerve II (optic nerve) segmentation from multimodal MRI. The dataset comprises 151 subjects from the Human Connectome Project (HCP) Young Adult cohort, including high-resolution T1-weighted structural MRI, multi-shell diffusion MRI, and high-quality binary segmentation masks of the bilateral cranial nerve II, optic chiasm, and optic tracts.

This repository contains all code for the annotation pipeline, preprocessing scripts, and baseline semi-supervised (Stage 1, LeFeD) and fully supervised (3D U-Net) segmentation methods described in our paper:

> Diakite, A., Li, C., Yu, S., Xie, L., Feng, Y., Zou, J. & Wang, S. OpticNerveSeg: A Public Dataset for Automatic Segmentation of the Cranial Nerve II from Multimodal MRI. *Scientific Data* (under review).

## 🎯 Key Features

- **151 subjects** with high-quality binary masks
- **Multimodal data**: T1w structural MRI + multi-shell diffusion MRI (b=1000,2000,3000 s/mm²)
- **Anatomical coverage**: cranial nerve II including optic nerve, optic chiasm, and optic tracts
- **Rich metadata**: mean radius, curvature, length, SNR, contrast, partial volume indices
- **Human-in-the-loop annotations**: 18 expert-annotated subjects (tractography-based seed annotations) + semi-supervised inference on 151 subjects + expert correction on 20 flagged cases
- **75% reduction** in expert annotation time compared to full manual segmentation
- **7 semi-supervised baselines** (mean teacher, uncertainty-aware mean teacher, UGMCL, entropy minimization, ICT, URPC, LeFeD) plus a fully supervised 3D U-Net baseline, with configurable labeled/unlabeled splits
- **Pretrained checkpoints included** for the LeFeD (Stage 1) and 3D U-Net (supervised baseline) models, 5-fold each, along with the exact train/test split manifests used

## 📊 Dataset Access

The dataset is available on BALSA: [https://balsa.wustl.edu/study/N9G6D](https://balsa.wustl.edu/study/N9G6D)

**Contents:**
- 151 binary masks and T1w images (`.nii.gz`) at 1.25 mm isotropic resolution
- Metadata CSV with subject-level anatomical and image quality features

**Note:** Raw HCP images are not redistributed due to data use terms. Users must obtain the original S1200 data from [ConnectomeDB](https://db.humanconnectome.org).

## 🚀 Quick Start

### Prerequisites

- [MRtrix3](https://www.mrtrix.org/) and [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki) (for the tractography/preprocessing scripts)
- Python 3.8+

### Installation

```bash
git clone https://github.com/aldiak/OpticNerveSeg.git
cd OpticNerveSeg
pip install -r scripts/requirements.txt
```
## Training and Inference

```
For training, download the data and change the path in run_all_alg.py and run it.
For Testing, change the data path and run infer.py.

The stage I test data can be obtained from us (Email: aldiak95@gmail.com) before training and inference to get Table I results.
The final data downloaded from BAlSA needs to be converted to H5 before training and inference to get Table II results.
```

## 📁 Repository Structure

```
OpticNerveSeg/
├── metadata/                  # Subject-level metadata
│   └── subjects_metadata.csv
├── scripts/                   # All code
│   ├── requirements.txt       # Python dependencies
│   ├── tractography/          # MRtrix3 tractography pipeline
│   │   ├── 01_dwi_tensor.sh   # Generates FA from the raw DWI
│   │   └── 02_tractography.sh # See also: https://github.com/IPIS-XieLei/CNsAtlas
│   └── semi_supervised/       # Semi-supervised + supervised baseline methods
│       ├── dataloaders/
│       ├── data_split/        # Train/test subject manifests (131/20 and Stage 1's 92/10 splits)
│       ├── networks/
│       ├── utils/
│       ├── model/             # Pretrained checkpoints (LeFeD and 3D U-Net, 5-fold each)
│       ├── evaluate.py
│       ├── test_util.py
│       ├── infer.py           # Set the paths correctly and run
│       └── run_all_alg.py     # Training entry point (--method selects the SSL algorithm)
├── LICENSE                    # MIT License
└── README.md                  # This file
```

Masks and T1w images themselves are not stored in this repository — they are distributed via BALSA (see [Dataset Access](#-dataset-access) above).
