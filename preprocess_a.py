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
    gray = np.full_like(img, 127, dtype=np.uint8)
    top_h = int(h * 0.4)
    gray[:top_h, :, :] = img[:top_h, :, :]
    img = gray

    # shrink to 90% and center with gray padding so later transforms don't crop edges
    scale = 0.90
    scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    pad = np.full_like(img, 127, dtype=np.uint8)
    sh, sw = scaled.shape[:2]
    y_off = (h - sh) // 2
    x_off = (w - sw) // 2
    pad[y_off:y_off + sh, x_off:x_off + sw] = scaled
    img = pad

    # random rotation and translation while keeping canvas size
    angle = random.uniform(-8, 8)
    tx = random.randint(-10, 10)
    ty = random.randint(-10, 10)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))
    cv2.imwrite(output_path, img)

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
