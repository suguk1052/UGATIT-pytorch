import argparse
import json
import os
import random
from typing import List

from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms
import numpy as np
try:
    from scipy import linalg as scipy_linalg
    _HAVE_SCIPY = True
except Exception:  # scipy may not be installed
    scipy_linalg = None
    _HAVE_SCIPY = False

try:
    Identity = nn.Identity  # available in newer PyTorch versions
except AttributeError:
    class Identity(nn.Module):
        """Fallback identity layer for older PyTorch releases."""
        def forward(self, x):
            return x


def list_image_files(directory: str) -> List[str]:
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [os.path.join(directory, f) for f in os.listdir(directory)
             if f.lower().endswith(exts)]
    return sorted(files)



def load_inception(device: str):
    """Load InceptionV3 model and the associated preprocessing transforms.

    When running on newer ``torchvision`` versions, the function uses the
    ``Inception_V3_Weights`` utility so that inputs are normalized exactly as
    expected by the pretrained weights. On older releases, it falls back to
    ``pretrained=True`` and a manual transform matching common practice.
    """
    try:
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        model = models.inception_v3(weights=weights)
        transform = weights.transforms()
    except AttributeError:  # older torchvision
        model = models.inception_v3(pretrained=True, transform_input=True)
        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
        ])
    model.fc = Identity()
    model.eval()
    model.to(device)
    return model, transform


def extract_features(paths: List[str], model: nn.Module, device: str,
                     batch_size: int, transform) -> torch.Tensor:
    images = []
    for p in paths:
        img = Image.open(p).convert('RGB')
        images.append(transform(img))
    loader = torch.utils.data.DataLoader(images, batch_size=batch_size,
                                         shuffle=False, drop_last=False)
    feats = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            feat = model(batch)
            feats.append(feat.cpu())
    return torch.cat(feats, dim=0)


def compute_kid(feats_fake: torch.Tensor, feats_real: torch.Tensor) -> float:
    m = feats_fake.size(0)
    n = feats_real.size(0)
    d = feats_fake.size(1)
    c = 1.0
    g = 1.0 / d
    k_xx = (g * feats_fake @ feats_fake.t() + c).pow(3)
    k_yy = (g * feats_real @ feats_real.t() + c).pow(3)
    k_xy = (g * feats_fake @ feats_real.t() + c).pow(3)
    sum_k_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (m * (m - 1))
    sum_k_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (n * (n - 1))
    sum_k_xy = k_xy.sum() / (m * n)
    kid = sum_k_xx + sum_k_yy - 2 * sum_k_xy
    return kid.item()


def _covariance(feats: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """Compute the unbiased covariance of a feature tensor."""
    diff = feats - mean
    return diff.t().mm(diff) / (feats.size(0) - 1)


def _sqrtm(mat: torch.Tensor) -> torch.Tensor:
    """Matrix square root using SciPy when available or an eigen fallback."""
    m = mat.cpu().numpy().astype(np.float64)
    if _HAVE_SCIPY:
        sqrt_m, _ = scipy_linalg.sqrtm(m, disp=False)
        if np.iscomplexobj(sqrt_m):
            sqrt_m = sqrt_m.real
    else:
        vals, vecs = np.linalg.eig(m)
        vals = np.clip(vals.real, a_min=0, a_max=None)
        sqrt_m = vecs @ np.diag(np.sqrt(vals)) @ np.linalg.inv(vecs)
        if np.iscomplexobj(sqrt_m):
            sqrt_m = sqrt_m.real
    return torch.from_numpy(sqrt_m).to(mat.device)


def compute_fid(feats_fake: torch.Tensor, feats_real: torch.Tensor) -> float:
    mu_fake = feats_fake.mean(dim=0)
    mu_real = feats_real.mean(dim=0)
    cov_fake = _covariance(feats_fake, mu_fake)
    cov_real = _covariance(feats_real, mu_real)

    diff = mu_fake - mu_real
    cov_prod = cov_fake.mm(cov_real)
    cov_sqrt = _sqrtm(cov_prod)
    fid = diff.dot(diff) + torch.trace(cov_fake + cov_real - 2 * cov_sqrt)
    fid_val = fid.item()
    if fid_val < 0:
        fid_val = 0.0
    return fid_val


def main():
    parser = argparse.ArgumentParser(
        description='Compute KID and FID on UGATIT results'
    )
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--direction', choices=['A2B', 'B2A'], default='A2B',
                        help='Translation direction to evaluate')
    parser.add_argument('--dataset_root', default='dataset',
                        help='Root directory for datasets')
    parser.add_argument('--real_dir', default=None,
                        help='Optional directory of real images to use as ground truth')
    parser.add_argument('--result_dir', default='results',
                        help='Directory containing generated results')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of images to evaluate')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output', default=None,
                        help='Optional path to save the result JSON')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.real_dir is None:
        real_dir = os.path.join(
            args.dataset_root,
            args.dataset,
            'testB' if args.direction == 'A2B' else 'testA'
        )
    else:
        real_dir = args.real_dir
    fake_dir = os.path.join(args.result_dir, args.dataset, 'test', args.direction)

    real_paths = list_image_files(real_dir)
    fake_paths = list_image_files(fake_dir)

    if not fake_paths:
        raise RuntimeError(
            f'No generated images found in {fake_dir}. '
            f'Run "python main.py --dataset {args.dataset} --phase test" first.'
        )

    if len(real_paths) < args.num_samples or len(fake_paths) < args.num_samples:
        raise ValueError('Not enough images for the requested number of samples')
    random.shuffle(real_paths)
    random.shuffle(fake_paths)
    real_paths = real_paths[:args.num_samples]
    fake_paths = fake_paths[:args.num_samples]

    print(f'Computing KID and FID for {args.dataset} {args.direction} on '
          f'{args.num_samples} images...')

    model, transform = load_inception(device)

    feats_real = extract_features(real_paths, model, device, args.batch_size, transform)
    feats_fake = extract_features(fake_paths, model, device, args.batch_size, transform)

    kid_score = compute_kid(feats_fake, feats_real)
    fid_score = compute_fid(feats_fake, feats_real)
    kid_x100 = kid_score * 100
    print(
        f'KID score for {args.dataset} {args.direction} '
        f'(mean over {args.num_samples} images): '
        f'{kid_score:.6f} ({kid_x100:.4f} x100)'
    )
    print(
        f'FID score for {args.dataset} {args.direction} '
        f'(mean over {args.num_samples} images): '
        f'{fid_score:.6f}'
    )

    if args.output is None:
        out_dir = os.path.join(args.result_dir, args.dataset, 'eval')
        os.makedirs(out_dir, exist_ok=True)
        kid_output = os.path.join(out_dir, f'kid_score_{args.direction}.json')
    else:
        kid_output = args.output
        out_dir = os.path.dirname(args.output)
    fid_output = os.path.join(out_dir, f'fid_score_{args.direction}.json')
    with open(kid_output, 'w') as f:
        json.dump(
            {
                'dataset': args.dataset,
                'direction': args.direction,
                'kid': kid_score,
                'kid_x100': kid_x100,
                'num_samples': args.num_samples,
            },
            f,
            indent=2,
        )

    with open(fid_output, 'w') as f:
        json.dump(
            {
                'dataset': args.dataset,
                'direction': args.direction,
                'fid': fid_score,
                'num_samples': args.num_samples,
            },
            f,
            indent=2,
        )


if __name__ == '__main__':
    main()
