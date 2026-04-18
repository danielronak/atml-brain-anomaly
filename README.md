# Unsupervised 3D Brain Anomaly Detection
**Team:** Ronak Daniel · Neel Jaiswal  
**Course:** Advanced Topics in Machine Learning (ATML)  
**Dataset:** BraTS 2021 (test) · IXI (healthy training)

---

## Project Overview

Unsupervised anomaly detection in 3D brain MRI using reconstruction-based
generative models. A model trained on **healthy anatomy** (IXI dataset, ~600
subjects) learns to reconstruct only healthy structure. At inference time,
pathological regions (gliomas) are not well-reconstructed → the pixel-wise
residual becomes the anomaly map.

We compare **4 architectures** across varying training methodology and
data quality:

| # | Model | Generator | Loss | Training Data | Resolution |
|---|---|---|---|---|---|
| 1 | CNN GAN (Naive) | 3D CNN | BCE | BraTS "healthy" slices | 64³ |
| 2 | CNN GAN (Proper) | 3D CNN | WGAN-GP | IXI healthy (T1+T2) | 128³ |
| 3 | Swin-UNET GAN | MONAI SwinUNETR | WGAN-GP | IXI healthy (T1+T2) | 128³ |
| 4 | VQ-VAE ⭐ | VQ-VAE | Recon+Codebook | IXI healthy (T1+T2) | 128³ |

Model 1 is the documented baseline failure. Models 2–4 are the rebuilt pipeline.

---

## Repository Structure

```
atml-brain-anomaly/
│
├── configs/
│   └── default.yaml              ← All hyperparameters (edit this first)
│
├── data/
│   ├── download_ixi.py           ← Download IXI T1+T2 to Drive
│   ├── raw/                      ← BraTS 2021 Training Data (local only)
│   └── processed/                ← Preprocessed cache
│
├── src/
│   ├── data/
│   │   └── dataset.py            ← IXI + BraTS dual-modality dataloaders
│   ├── models/
│   │   ├── baseline.py           ← CNN naive: Generator3D, Discriminator3D
│   │   ├── encoder.py            ← izi_f encoder (for GAN models)
│   │   ├── vqvae.py              ← Model 4: VQ-VAE (MONAI)
│   │   ├── swin_generator.py     ← Model 3: SwinUNETR generator
│   │   └── patch_discriminator.py← Shared PatchGAN discriminator
│   ├── training/
│   │   ├── train_vqvae.py        ← VQ-VAE trainer
│   │   └── train_gan.py          ← Unified WGAN-GP trainer (Models 2+3)
│   └── evaluation/
│       ├── metrics.py            ← Dice, AUROC, Hausdorff95
│       └── anomaly_scorer.py     ← Unified inference for all models
│
├── notebooks/                    ← Run these in Google Colab (Ronak)
│   ├── 01_data_preparation.ipynb ← Download IXI, verify BraTS
│   ├── 02_train_vqvae.ipynb      ← Runtime 1: VQ-VAE training (~5-6 hrs)
│   ├── 03_train_swin_gan.ipynb   ← Runtime 2: Swin GAN training (~6-8 hrs)
│   ├── 04_train_cnn_proper.ipynb ← Optional: CNN Proper
│   └── 05_evaluation.ipynb       ← All-model comparison + figures
│
├── results/                      ← Model outputs (large files gitignored)
│   ├── cnn_naive/                ← Model 1 metrics + anomaly maps
│   ├── cnn_proper/               ← Model 2 metrics + anomaly maps
│   ├── swin_gan/                 ← Model 3 metrics + anomaly maps
│   └── vqvae/                    ← Model 4 metrics + anomaly maps
│
├── baseline_cnn/                 ← Phase 1–3: Original CNN pipeline (documented)
│   ├── checkpoints/              ← Trained weights (Generator, Discriminator, Encoder)
│   ├── results/                  ← Loss curves (gan_loss_curve.png, encoder_loss_curve.png)
│   └── scripts/                  ← Original training scripts (train_baseline.py, etc.)
│
├── docs/
│   ├── Project Status and Architectural Analysis Report.docx
│   └── paper/figures/            ← Figures for the final report
│
├── archive/                      ← Historical dead-ends, documented
│   ├── azure_plan/               ← Original Azure ML pipeline (GPU quota denied)
│   ├── 2d_model_drafts/          ← Early 2D Conv experiments
│   ├── trainer.py                ← WGAN-GP trainer for Azure (create_graph=True fails on MPS)
│   ├── preprocess.py             ← Old 2D slice extractor
│   └── gan_losses.py             ← Old standalone WGAN losses (absorbed into train_gan.py)
│
├── check_compute.py              ← Quick GPU/MPS hardware check
├── environment.yml               ← Conda environment
└── requirements.txt
```

---

## Quick Start (Colab)

```python
# ── In each Colab notebook, Run Cell 1 ────────────────────────
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/YOUR_USER/atml-brain-anomaly /content/atml
%cd /content/atml
!pip install -q monai monai-generative einops nibabel pyyaml scipy pandas

# ── Update Drive paths in config ──────────────────────────────
import yaml
with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)
config['data']['ixi_dir']        = '/content/drive/MyDrive/atml/data/ixi'
config['data']['brats_dir']      = '/content/drive/MyDrive/atml/data/brats2021'
config['data']['checkpoint_dir'] = '/content/drive/MyDrive/atml/checkpoints'
config['data']['results_dir']    = '/content/drive/MyDrive/atml/results'
```

**Notebook order:**
1. Run `01_data_preparation.ipynb` once (no GPU needed)
2. Open two runtimes simultaneously:
   - **Runtime 1:** `02_train_vqvae.ipynb`
   - **Runtime 2:** `03_train_swin_gan.ipynb`
3. After both complete: `05_evaluation.ipynb`

---

## Key Design Decisions

**Why IXI for training?**  
BraTS contains only glioma patients. Training on BraTS "healthy slices"
contaminates the model's definition of normal with tumour-adjacent anatomy.
IXI provides ~600 truly healthy volunteers.

**Why dual T1+T2?**  
Training on both modalities jointly allows the model to learn inter-modal
correlations. Tumours violate both within-modality statistics AND T1/T2
co-occurrence patterns, providing a richer anomaly signal.

**Why VQ-VAE?**  
GANs require careful adversarial balancing. VQ-VAE's discrete codebook
constrains reconstructions to healthy anatomy tokens without an adversarial
training game — eliminating mode collapse risk entirely.

**Why WGAN-GP for GANs?**  
Standard BCE GANs collapsed at epoch 40 in the baseline (Model 1).
WGAN-GP enforces Lipschitz continuity via gradient penalty, providing
stable gradient feedback. Works on CUDA (Colab); fails on MPS (hence M5 failure).