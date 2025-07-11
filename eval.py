import argparse
import json
import os
import random
from typing import List

from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms

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


def list_image_files_prefix(directory: str, prefix: str) -> List[str]:
    """List image files with a specific prefix inside a directory."""
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [os.path.join(directory, f) for f in os.listdir(directory)
             if f.startswith(prefix) and f.lower().endswith(exts)]
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


def main():
    parser = argparse.ArgumentParser(description='Compute KID on UGATIT results')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--direction', choices=['A2B', 'B2A'], default='A2B',
                        help='Translation direction to evaluate')
    parser.add_argument('--dataset_root', default='dataset',
                        help='Root directory for datasets')
    parser.add_argument('--result_dir', default='results',
                        help='Directory containing generated results')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of images to evaluate')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output', default=None,
                        help='Optional path to save the result JSON')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    real_dir = os.path.join(args.dataset_root, args.dataset,
                            'testB' if args.direction == 'A2B' else 'testA')
    fake_dir = os.path.join(args.result_dir, args.dataset, 'test')

    prefix = args.direction + '_'
    real_paths = list_image_files(real_dir)
    fake_paths = list_image_files_prefix(fake_dir, prefix)

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

    print(f'Computing KID for {args.dataset} {args.direction} on '
          f'{args.num_samples} images...')

    model, transform = load_inception(device)

    feats_real = extract_features(real_paths, model, device, args.batch_size, transform)
    feats_fake = extract_features(fake_paths, model, device, args.batch_size, transform)

    kid_score = compute_kid(feats_fake, feats_real)
    kid_x100 = kid_score * 100
    print(
        f'KID score for {args.dataset} {args.direction} '
        f'(mean over {args.num_samples} images): '
        f'{kid_score:.6f} ({kid_x100:.4f} x100)'
    )

    if args.output is None:
        out_dir = os.path.join(args.result_dir, args.dataset, 'eval')
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(out_dir, f'kid_score_{args.direction}.json')

    with open(args.output, 'w') as f:
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


if __name__ == '__main__':
    main()
