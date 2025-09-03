import os
import argparse
import random

def process_image(img_path, output_path, crop_mode=None):
    import cv2
    import numpy as np

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Unable to read {img_path}")
        return

    h, w = img.shape[:2]
    extra_w = int(round(w * 0.15))
    extra_h = int(round(h * 0.15))
    gray_val = 128
    gray = (gray_val, gray_val, gray_val)

    if crop_mode == "bottom":
        # keep the bottom 30% and fade between 65% and 75%
        mask = np.zeros((h, w), dtype=np.float32)
        mask[int(h * 0.7) :, :] = 1.0
        sigma = h * 0.05 / 3.0  # ~10% height transition zone
        mask = cv2.GaussianBlur(mask, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
        mask[: int(h * 0.65), :] = 0.0
        mask[int(h * 0.75) :, :] = 1.0
        center_ratio = 0.93
    elif crop_mode == "top":
        # keep the top 40% and fade between 35% and 45%
        mask = np.zeros((h, w), dtype=np.float32)
        mask[: int(h * 0.4), :] = 1.0
        sigma = h * 0.05 / 3.0  # ~10% height transition zone
        mask = cv2.GaussianBlur(mask, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
        mask[: int(h * 0.35), :] = 1.0
        mask[int(h * 0.45) :, :] = 0.0
        center_ratio = 0.2
    else:
        mask = None

    if mask is not None:
        gray_img = np.full_like(img, gray_val, dtype=np.uint8)
        blended = img.astype(np.float32) * mask[:, :, None] + gray_img.astype(np.float32) * (1 - mask[:, :, None])
        blended = blended.astype(np.uint8)
    else:
        blended = img

    # compute padding so rotation/translation don't crop the image
    max_trans = 10
    max_angle = 10
    rad = np.deg2rad(max_angle)
    w_rot = w * abs(np.cos(rad)) + h * abs(np.sin(rad))
    h_rot = h * abs(np.cos(rad)) + w * abs(np.sin(rad))
    pad_w = int(np.ceil((w_rot - w) / 2 + max_trans))
    pad_h = int(np.ceil((h_rot - h) / 2 + max_trans))

    if crop_mode == "bottom":
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
    transformed = cv2.warpAffine(
        padded, M, (pw, ph), borderMode=cv2.BORDER_CONSTANT, borderValue=gray
    )

    extra_left = extra_w if tx > 0 else 0
    extra_right = extra_w if tx < 0 else 0
    extra_top = extra_h if ty < 0 else 0
    extra_bottom = extra_h if ty > 0 else 0
    if extra_left or extra_right or extra_top or extra_bottom:
        transformed = cv2.copyMakeBorder(
            transformed,
            extra_top,
            extra_bottom,
            extra_left,
            extra_right,
            cv2.BORDER_CONSTANT,
            value=gray,
        )

    th, tw = transformed.shape[:2]
    if crop_mode is None:
        cv2.imwrite(output_path, transformed)
        return

    # scale to cover 512x512 and crop so a reference band sits at the canvas center
    target = 512
    scale = target / min(tw, th)
    if crop_mode == "bottom":
        scale *= 1.1
    new_w = int(np.ceil(tw * scale))
    new_h = int(np.ceil(th * scale))
    resized = cv2.resize(transformed, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    base_y = (new_h - target) // 2
    bias = int((0.5 - center_ratio) * h * scale)
    y_off = min(max(base_y - bias, 0), new_h - target)
    x_off = max((new_w - target) // 2, 0)
    crop = resized[y_off : y_off + target, x_off : x_off + target]
    cv2.imwrite(output_path, crop)

def process_dir(source_dir, dest_dir, crop_mode=None):
    os.makedirs(dest_dir, exist_ok=True)
    for fname in os.listdir(source_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            src = os.path.join(source_dir, fname)
            base = os.path.splitext(fname)[0]
            dst = os.path.join(dest_dir, base + ".png")
            process_image(src, dst, crop_mode)

def main():
    parser = argparse.ArgumentParser(description="Preprocess images with optional cropping.")
    parser.add_argument("--input_dir", required=True, help="Directory containing raw images.")
    parser.add_argument("--output_dir", required=True, help="Directory to save processed images.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--top", action="store_true", help="keep top 40%% and fade between 35%% and 45%%")
    group.add_argument("--bottom", action="store_true", help="keep bottom 30%% and fade between 65%% and 75%%")
    args = parser.parse_args()

    if args.bottom:
        mode = "bottom"
    elif args.top:
        mode = "top"
    else:
        mode = None

    process_dir(args.input_dir, args.output_dir, mode)

if __name__ == "__main__":
    main()
