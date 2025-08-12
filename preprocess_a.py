import os
import argparse
import random
import cv2
import numpy as np

def process_image(img_path, output_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Unable to read {img_path}")
        return

    h, w = img.shape[:2]
    gray_val = 127

    # build mask keeping the top 40% but fading out between 35% and 45%
    mask = np.zeros((h, w), dtype=np.float32)
    mask[: int(h * 0.4), :] = 1.0
    sigma = h * 0.05 / 3.0  # ~10% height transition zone
    mask = cv2.GaussianBlur(mask, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    mask[: int(h * 0.35), :] = 1.0
    mask[int(h * 0.45) :, :] = 0.0
    gray_img = np.full_like(img, gray_val, dtype=np.uint8)
    blended = img.astype(np.float32) * mask[:, :, None] + gray_img.astype(np.float32) * (1 - mask[:, :, None])
    blended = blended.astype(np.uint8)

    # compute padding so rotation/translation don't crop the image
    max_trans = 10
    max_angle = 10
    rad = np.deg2rad(max_angle)
    w_rot = w * abs(np.cos(rad)) + h * abs(np.sin(rad))
    h_rot = h * abs(np.cos(rad)) + w * abs(np.sin(rad))
    pad_w = int(np.ceil((w_rot - w) / 2 + max_trans))
    pad_h = int(np.ceil((h_rot - h) / 2 + max_trans))
    gray = (gray_val, gray_val, gray_val)
    padded = cv2.copyMakeBorder(
        blended, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=gray
    )

    ph, pw = padded.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    tx = random.uniform(-max_trans, max_trans)
    ty = random.uniform(-max_trans, max_trans)
    M = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    transformed = cv2.warpAffine(
        padded, M, (pw, ph), borderMode=cv2.BORDER_CONSTANT, borderValue=gray
    )

    # scale to cover 512x512 and crop the center
    target = 512
    scale = target / min(pw, ph)
    new_w = int(np.ceil(pw * scale))
    new_h = int(np.ceil(ph * scale))
    resized = cv2.resize(transformed, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    y_off = max((new_h - target) // 2, 0)
    x_off = max((new_w - target) // 2, 0)
    crop = resized[y_off:y_off + target, x_off:x_off + target]
    cv2.imwrite(output_path, crop)

def process_split(source_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for fname in os.listdir(source_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            src = os.path.join(source_dir, fname)
            base = os.path.splitext(fname)[0]
            dst = os.path.join(dest_dir, base + ".png")
            process_image(src, dst)

def main():
    parser = argparse.ArgumentParser(description="Preprocess domain A images.")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset inside dataset/.")
    parser.add_argument("--source_root", default="preprocess_source", help="Root directory containing raw images.")
    parser.add_argument("--dataset_root", default="dataset", help="Root directory for processed dataset.")
    args = parser.parse_args()
    for split in ["trainA", "testA"]:
        src_dir = os.path.join(args.source_root, split)
        dst_dir = os.path.join(args.dataset_root, args.dataset_name, split)
        process_split(src_dir, dst_dir)

if __name__ == "__main__":
    main()
