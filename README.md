# U-GAT-IT (PyTorch)

This fork of [znxlwm/UGATIT-pytorch](https://github.com/znxlwm/UGATIT-pytorch) provides a streamlined setup for training, testing and computing KID scores.

## Setup

Install the requirements:

```bash
pip install -r requirements.txt
```

## Dataset Layout

```
dataset/
  YOUR_DATASET_NAME/
    trainA/
      *.jpg|png
    trainB/
      *.jpg|png
    testA/
      *.jpg|png
    testB/
      *.jpg|png
```

## Train

```bash
python main.py --dataset YOUR_DATASET_NAME
```

Add `--light` if GPU memory is limited. For rectangular images you can use `--aspect_ratio <width/height>`.

## Test

```bash
python main.py --dataset YOUR_DATASET_NAME --phase test
```

## KID Evaluation

After running the test phase, compute the score as follows:

```bash
python eval.py --dataset YOUR_DATASET_NAME --direction A2B --num_samples 100
```

Real images are read from `dataset/YOUR_DATASET_NAME/testB` (or `testA` for `B2A`) and generated images from `results/YOUR_DATASET_NAME/test`. The result is saved to `results/YOUR_DATASET_NAME/eval/kid_score_<direction>.json`.
