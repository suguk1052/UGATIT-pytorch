## U-GAT-IT &mdash; Official PyTorch Implementation
### : Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation

> **Note:** This repo is a fork of [znxlwm/UGATIT-pytorch](https://github.com/znxlwm/UGATIT-pytorch), with modifications for personal experiments and research. See below for environment setup.

<div align="center">
  <img src="./assets/teaser.png">
</div>

### [Paper](https://arxiv.org/abs/1907.10830) | [Official Tensorflow code](https://github.com/taki0112/UGATIT)
The results of the paper came from the **Tensorflow code**


> **U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation**<br>
>
> **Abstract** *We propose a novel method for unsupervised image-to-image translation, which incorporates a new attention module and a new learnable normalization function in an end-to-end manner. The attention module guides our model to focus on more important regions distinguishing between source and target domains based on the attention map obtained by the auxiliary classifier. Unlike previous attention-based methods which cannot handle the geometric changes between domains, our model can translate both images requiring holistic changes and images requiring large shape changes. Moreover, our new AdaLIN (Adaptive Layer-Instance Normalization) function helps our attention-guided model to flexibly control the amount of change in shape and texture by learned parameters depending on datasets. Experimental results show the superiority of the proposed method compared to the existing state-of-the-art models with a fixed network architecture and hyper-parameters.*

## Usage
```
├── dataset
   └── YOUR_DATASET_NAME
       ├── trainA
            ├── xxx.jpg (name, format doesn't matter)
            ├── yyy.png
            └── ...
       ├── trainB
            ├── zzz.jpg
            ├── www.png
            └── ...
       ├── testA
            ├── aaa.jpg
            ├── bbb.png
            └── ...
       └── testB
            ├── ccc.jpg
            ├── ddd.png
            └── ...
```

### Domain A preprocessing (optional)

If you need to crop and augment domain A images before training, place the
original files under `preprocess_source/trainA` and `preprocess_source/testA`
and run:

```
python preprocess_a.py --dataset_name YOUR_DATASET_NAME [--bottom]
```

Run this once before starting `main.py` to prepare domain A images.

By default the script keeps only the top 40 % of each image, fading to neutral
gray across the 35–45 % band using a Gaussian kernel. With the `--bottom` flag,
it instead retains the bottom 30 %, fading the 65–75 % band. The resulting
canvas undergoes random translations up to ±10 pixels and random rotations up to
±10°, with any exposed regions filled with the same gray. The output is scaled
to cover a 512×512 frame while keeping aspect ratio and cropped so the row at
20 % (or 93 % when using `--bottom`) of the original height falls at the canvas
center. Each processed file keeps the original base name and is written to
`dataset/YOUR_DATASET_NAME/trainA` and `dataset/YOUR_DATASET_NAME/testA`. After
running this preprocessing step you can proceed with the usual training command
below.

### Train
```
> python main.py --dataset selfie2anime
```
* If the memory of gpu is **not sufficient**, set `--light` to True
* Enable style diversity with `--use_ds` (see `--style_dim` and `--ds_weight`)
* Inject reference-domain style and CAM-weighted losses with `--use_spade_adalin`
  (tune `--style_nc`, `--lambda_style`, `--lambda_lowpass`, `--lambda_highpass`,
  `--fg_bg_cycle_ratio`)
* Save memory with `--use_checkpoint` for gradient checkpointing
* To train with a rectangular resolution, set `--aspect_ratio <width/height>`.
  The resulting width `img_size * aspect_ratio` must be divisible by 4.
  For example, a 1:2.3 ratio is approximated with `--aspect_ratio 0.44` when
  using the default `--img_size 256`.
* Adjust global vs. local discriminator losses with `--global_dis_ratio <0~1>`.
  The local ratio is `1 - global_dis_ratio`.

### Test
```
> python main.py --dataset selfie2anime --phase test
```
* Use `--resume_iter N` to load a specific checkpoint during testing.
* Generated images are stored under `results/YOUR_DATASET_NAME/test/A2B/` and
  `test/B2A/` using the original filenames from `testA` and `testB`.

## Architecture
<div align="center">
  <img src = './assets/generator.png' width = '785px' height = '500px'>
</div>

---

<div align="center">
  <img src = './assets/discriminator.png' width = '785px' height = '450px'>
</div>

## Results
### Ablation study
<div align="center">
  <img src = './assets/ablation.png' width = '438px' height = '346px'>
</div>

### User study
<div align="center">
  <img src = './assets/user_study.png' width = '738px' height = '187px'>
</div>

### Comparison
<div align="center">
  <img src = './assets/kid.png' width = '787px' height = '344px'>
</div>

---

## 🛠️ Key Modifications

* `--aspect_ratio` option for non-square training resolutions
* `--global_dis_ratio` to balance global and local discriminators
* Optional style diversity via `--use_ds`, `--style_dim`, and `--ds_weight`
* Gradient checkpointing via `--use_checkpoint`
* Optional center cropping via `--center_crop` to keep aspect ratio without padding
* SPADE-AdaLIN style conditioning (`--use_spade_adalin`, `--style_nc`, `--lambda_style`,
  `--lambda_lowpass`, `--lambda_highpass`, `--fg_bg_cycle_ratio`) that splits a random
  domain B reference into CAM-based foreground/background style codes, copies the source
  background during A→B→A cycles, penalizes background high-frequency mismatch, weights
  cycle/identity losses with the source-domain masks, and limits skip connections to the
  foreground to curb background bleeding

## 🛠️ Local Setup (for forked repo by @suguk1052)

This fork includes minor modifications and experiments.
If you'd like to test or reproduce the results, follow the setup below.

### ✅ Conda-based Environment Setup

```bash
conda create -n ugatit python=3.6.9 -y
conda activate ugatit

git clone https://github.com/suguk1052/UGATIT-pytorch.git
cd UGATIT-pytorch

pip install -r requirements.txt
```

### KID / FID Evaluation
After running `main.py --phase test`, compute the scores:
```bash
python eval.py --dataset YOUR_DATASET_NAME --direction A2B
```
Add `--num_samples N` to limit the evaluation to at most `N` images from each
directory. By default all available images are used with no random shuffling.
Results are written to `results/YOUR_DATASET_NAME/eval/` by default. Use
`--result_dir OTHER_DIR` if your generated images live elsewhere. The
script creates `kid_score_A2B.json` and `fid_score_A2B.json` (or
`*_B2A.json` when evaluating the opposite direction). The metrics are
computed from the images saved under `results/YOUR_DATASET_NAME/test/A2B/`
or `test/B2A/`. Each JSON also records how many real and fake images were
used. The KID file contains both the raw `kid` value and `kid_x100` for
convenience.
You can also specify `--real_dir PATH` to override the directory of real
images used as ground truth (by default it is `testB` or `testA`).
