#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import os
import json
import lmdb
import io
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    try:
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return False
        return True
    except:
        return False


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def collect_annotations_from_root(root_dir):
    """
    [ {"image_path": ..., "label": "..."} ]
    """
    all_samples = []
    class_dirs = sorted(os.listdir(root_dir))
    for cls in class_dirs:
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        json_files = [f for f in os.listdir(cls_path) if f.endswith(".json")]
        if not json_files:
            print(f"⚠️{cls} no JSON ")
            continue
        for json_file in json_files:
            json_path = os.path.join(cls_path, json_file)
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for task_name, ann_list in data.get("annotations", {}).items():
                for ann in ann_list:
                    image_path = os.path.join(cls_path, ann["file_path"])
                    if not os.path.exists(image_path):
                        print(f"⚠️ image not exist: {image_path}")
                        continue
                    # all gt
                    texts = [g["text"] for g in ann.get("gt", [])]
                    label = "\t".join(texts)  
                    all_samples.append({
                        "image_path": image_path,
                        "label": label
                    })
    print(f"✅ from {root_dir} collect {len(all_samples)} images")
    return all_samples


def create_lmdb_dataset(samples, lmdb_path, map_size=1099511627776):
    os.makedirs(lmdb_path, exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=map_size)
    cache = {}
    cnt = 1
    valid_samples = 0

    print(f"start reading LMDB -> {lmdb_path}")
    for sample in tqdm(samples):
        image_path = sample["image_path"]
        label = sample["label"]

        try:
            with open(image_path, "rb") as f:
                imageBin = f.read()
        except Exception as e:
            print(f"❌ rading falis: {image_path}, error: {e}")
            continue

        if not checkImageIsValid(imageBin):
            print(f"⚠️ invalid picture: {image_path}")
            continue

        try:
            buf = io.BytesIO(imageBin)
            w, h = Image.open(buf).size
        except Exception:
            w, h = (0, 0)

        imageKey = ('image-%09d' % cnt).encode()
        labelKey = ('label-%09d' % cnt).encode()
        whKey = ('wh-%09d' % cnt).encode()

        cache[imageKey] = imageBin
        cache[labelKey] = label.encode("utf-8")
        cache[whKey] = f"{w}_{h}".encode("utf-8")

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print(f" write {cnt} images")

        cnt += 1
        valid_samples += 1

    if cache:
        writeCache(env, cache)

    with env.begin(write=True) as txn:
        txn.put(b"num-samples", str(valid_samples).encode())

    env.close()
    print(f"✅ LMDB done: {lmdb_path}, sample numbers: {valid_samples}")


def verify_lmdb(lmdb_path, num_check=3):
    print(f"\n validation LMDB dataset: {lmdb_path}")
    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin() as txn:
        num_samples = int(txn.get(b"num-samples").decode())
        print(f"Total number of samples: {num_samples}")
        for i in range(1, min(num_check + 1, num_samples + 1)):
            imageKey = ('image-%09d' % i).encode()
            labelKey = ('label-%09d' % i).encode()
            whKey = ('wh-%09d' % i).encode()
            label = txn.get(labelKey).decode()
            w, h = txn.get(whKey).decode().split("_")
            print(f"samole {i}: {label}, size: {w}x{h}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Convert OCR dataset to LMDB")
    parser.add_argument('--data_root', type=str,
                        default='The absolute path of SCUT-HCCDoc dataset',
                        help='Dataset root containing train/ and test/ folders')
    parser.add_argument('--output_dir', type=str,
                        default='The empty directory for saving images',
                        help='Output directory for LMDB')
    parser.add_argument('--map_size', type=int,
                        default=1099511627776,
                        help='LMDB map size (default: 1TB)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # train dataset
    train_root = os.path.join(args.data_root, "train")
    train_samples = collect_annotations_from_root(train_root)
    train_lmdb_path = os.path.join(args.output_dir, "train_lmdb")
    create_lmdb_dataset(train_samples, train_lmdb_path, args.map_size)
    verify_lmdb(train_lmdb_path)

    # test dataset
    test_root = os.path.join(args.data_root, "test")
    test_samples = collect_annotations_from_root(test_root)
    test_lmdb_path = os.path.join(args.output_dir, "test_lmdb")
    create_lmdb_dataset(test_samples, test_lmdb_path, args.map_size)
    verify_lmdb(test_lmdb_path)

    print("\n🎉 all done！")


if __name__ == "__main__":
    main()
