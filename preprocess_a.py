import os
import argparse
import random
import cv2
import numpy as np


def process_image(img_path, output_path, keep_bottom=False, seed=None, is_mask=False):
    if seed is not None:
        random.seed(seed)
    flag = cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR
    img = cv2.imread(img_path, flag)
    if img is None:
        print(f"Unable to read {img_path}")
        return

    h, w = img.shape[:2]
    gray_val = 1 if is_mask else 127

    if keep_bottom:
        # keep the bottom 30% and fade between 65% and 75%
        fade_mask = np.zeros((h, w), dtype=np.float32)
        fade_mask[int(h * 0.7) :, :] = 1.0
        sigma = h * 0.05 / 3.0  # ~10% height transition zone
        fade_mask = cv2.GaussianBlur(fade_mask, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
        fade_mask[: int(h * 0.65), :] = 0.0
        fade_mask[int(h * 0.75) :, :] = 1.0
        center_ratio = 0.93
    else:
        # keep the top 40% and fade between 35% and 45%
        fade_mask = np.zeros((h, w), dtype=np.float32)
        fade_mask[: int(h * 0.4), :] = 1.0
        sigma = h * 0.05 / 3.0  # ~10% height transition zone
        fade_mask = cv2.GaussianBlur(fade_mask, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
        fade_mask[: int(h * 0.35), :] = 1.0
        fade_mask[int(h * 0.45) :, :] = 0.0
        center_ratio = 0.2

    if is_mask:
        img = (img / 255.0).astype(np.float32)
        blended = img * fade_mask + 1.0 * (1 - fade_mask)
    else:
        gray_img = np.full_like(img, gray_val, dtype=np.uint8)
        blended = img.astype(np.float32) * fade_mask[:, :, None] + gray_img.astype(np.float32) * (1 - fade_mask[:, :, None])
    blended = blended.astype(np.float32 if is_mask else np.uint8)

    # compute padding so rotation/translation don't crop the image
    max_trans = 10
    max_angle = 10
    rad = np.deg2rad(max_angle)
    w_rot = w * abs(np.cos(rad)) + h * abs(np.sin(rad))
    h_rot = h * abs(np.cos(rad)) + w * abs(np.sin(rad))
    pad_w = int(np.ceil((w_rot - w) / 2 + max_trans))
    pad_h = int(np.ceil((h_rot - h) / 2 + max_trans))
    gray = gray_val
    if not is_mask:
        gray = (gray_val, gray_val, gray_val)

    if keep_bottom:
        top_pad = int(pad_h * 0.5)
        bottom_pad = int(pad_h * 1.5)
        side_pad = int(pad_w * 0.8)
        padded = cv2.copyMakeBorder(
            blended, top_pad, bottom_pad, side_pad, side_pad, cv2.BORDER_CONSTANT, value=gray
        )
    else:
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
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    transformed = cv2.warpAffine(
        padded, M, (pw, ph), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=gray
    )

    # scale to cover 512x512 and crop so a reference band sits at the canvas center
    target = 512
    scale = target / min(pw, ph)
    if keep_bottom:
        scale *= 1.1
    new_w = int(np.ceil(pw * scale))
    new_h = int(np.ceil(ph * scale))
    resized = cv2.resize(transformed, (new_w, new_h), interpolation=interp)
    base_y = (new_h - target) // 2
    bias = int((0.5 - center_ratio) * h * scale)
    y_off = min(max(base_y - bias, 0), new_h - target)
    x_off = max((new_w - target) // 2, 0)
    crop = resized[y_off : y_off + target, x_off : x_off + target]
    if is_mask:
        crop = (crop > 0.5).astype(np.uint8) * 255
    cv2.imwrite(output_path, crop)

def process_split(source_dir, dest_dir, mask_source_dir=None, mask_dest_dir=None, keep_bottom=False):
    os.makedirs(dest_dir, exist_ok=True)
    if mask_source_dir is not None:
        os.makedirs(mask_dest_dir, exist_ok=True)
    for fname in os.listdir(source_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            src = os.path.join(source_dir, fname)
            base = os.path.splitext(fname)[0]
            dst = os.path.join(dest_dir, base + ".png")
            seed = random.randint(0, 2 ** 32 - 1)
            process_image(src, dst, keep_bottom, seed, False)
            if mask_source_dir is not None:
                mask_src = os.path.join(mask_source_dir, fname)
                mask_dst = os.path.join(mask_dest_dir, base + ".png")
                process_image(mask_src, mask_dst, keep_bottom, seed, True)

def main():
    parser = argparse.ArgumentParser(description="Preprocess domain A images.")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset inside dataset/.")
    parser.add_argument("--source_root", default="preprocess_source", help="Root directory containing raw images.")
    parser.add_argument("--dataset_root", default="dataset", help="Root directory for processed dataset.")
    parser.add_argument("--bottom", action="store_true", help="keep bottom 30% instead of top 40%")
    args = parser.parse_args()
    for split in ["trainA", "testA"]:
        src_dir = os.path.join(args.source_root, split)
        dst_dir = os.path.join(args.dataset_root, args.dataset_name, split)
        mask_src = os.path.join(args.source_root, f"{split}_mask")
        mask_dst = os.path.join(args.dataset_root, args.dataset_name, f"{split}_mask")
        process_split(src_dir, dst_dir, mask_src, mask_dst, args.bottom)

if __name__ == "__main__":
    main()
