import os
import shutil
import zipfile
import json
import yaml
from pathlib import Path
from ultralytics import YOLO

# ================= Configuration =================
# set your local dataset zip directory
SRC_ZIP_DIR = Path("./Scout2map-Dataset").resolve()
WORK_DIR = Path("./workspace").resolve()
MERGED_DIR = WORK_DIR / "merged_dataset_7"
EXTRACT_DIR = WORK_DIR / "datasets"

# discrete 7 target classes with person_in_danger included
TARGET_CLASSES = [
    'person',
    'person_in_danger',
    'fire',
    'smoke',
    'exit_indicator',
    'gas_tank',
    'fire_extinguisher'
]

# all dataset zip archives including newly added ones
ZIP_FILES = [
    'Yolo-disaster-relief.zip',
    'exit-sign-Extended.zip',
    'fire-smoke-detection.zip',
    'gas-tank.zip',
    'fire-extinguisher.zip',
    'people.zip',
    'fire-detection-yolo.zip',
    'fire-smog.zip'
]

# precise rule mapping for each dataset folder to 7 target classes
def map_fire_smog(name):
    low = name.lower()
    if 'danger' in low:
        return 'person_in_danger'
    if 'smoke' in low or 'smog' in low:
        return 'smoke'
    if 'fire' in low or 'flame' in low:
        return 'fire'
    if 'person' in low or 'people' in low:
        return 'person'
    return None

def map_fire_detection_yolo(name):
    low = name.lower()
    if 'smoke' in low:
        return 'smoke'
    if 'fire' in low:
        return 'fire'
    return None

DATASET_RULES = {
    'exit-sign-Extended': lambda name: 'exit_indicator' if 'exit' in name.lower() else None,
    'fire-extinguisher': lambda name: 'fire_extinguisher' if 'extinguisher' in name.lower() else None,
    'gas-tank': lambda name: 'gas_tank' if 'gas' in name.lower() or 'tank' in name.lower() else None,
    'fire-smog': map_fire_smog,
    'fire-smoke-detection': lambda name: 'smoke' if 'smoke' in name.lower() else ('fire' if 'fire' in name.lower() else None),
    'fire-detection-yolo': map_fire_detection_yolo,
    'people': lambda name: 'person' if 'person' in name.lower() or 'people' in name.lower() else None,
    'Yolo-disaster-relief': lambda name: 'person' if 'person' in name.lower() else None,
}
# =================================================

def prepare_dataset():
    print("[1/4] Checking extracted datasets...")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    for zname in ZIP_FILES:
        zpath = SRC_ZIP_DIR / zname
        out_dir = EXTRACT_DIR / zpath.stem
        
        # unzip only if not extracted yet to save time
        if not out_dir.exists() and zpath.exists():
            print(f"  Extracting {zname}...")
            out_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zpath, 'r') as zf:
                zf.extractall(out_dir)

    print("[2/4] Parsing and converting datasets to merged_dataset_7 (7 classes)...")
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)

    for split in ['train', 'valid']:
        (MERGED_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

    class_counts = {cls_name: 0 for cls_name in TARGET_CLASSES}

    for folder in EXTRACT_DIR.iterdir():
        if not folder.is_dir():
            continue
        rule = DATASET_RULES.get(folder.name)
        if not rule:
            continue

        prefix = folder.name.replace('-', '_').lower()
        print(f"  Processing {folder.name}...")

        for split in ['train', 'valid', 'val', 'test']:
            split_dir = folder / split
            if not split_dir.exists():
                continue
            tgt_split = 'valid' if split in ['valid', 'val'] else 'train'

            out_img_dir = MERGED_DIR / tgt_split / 'images'
            out_lbl_dir = MERGED_DIR / tgt_split / 'labels'

            # 1. check for coco format (json files)
            json_files = list(split_dir.glob('*.json'))
            if json_files:
                with open(json_files[0], 'r', encoding='utf-8') as f:
                    coco = json.load(f)

                cat_id_to_idx = {}
                for c in coco.get('categories', []):
                    mapped_name = rule(c['name'])
                    if mapped_name in TARGET_CLASSES:
                        cat_id_to_idx[c['id']] = TARGET_CLASSES.index(mapped_name)

                images = {img['id']: img for img in coco.get('images', [])}
                img_to_anns = {img['id']: [] for img in coco.get('images', [])}
                for ann in coco.get('annotations', []):
                    img_id = ann['image_id']
                    if img_id in img_to_anns:
                        img_to_anns[img_id].append(ann)

                for img_id, img_info in images.items():
                    fname = img_info['file_name']
                    src_img = split_dir / fname
                    if not src_img.exists():
                        src_img = split_dir / 'images' / fname
                    if not src_img.exists():
                        continue

                    iw, ih = float(img_info['width']), float(img_info['height'])
                    new_stem = f"{prefix}_{Path(fname).stem}"
                    shutil.copy(src_img, out_img_dir / f"{new_stem}{Path(fname).suffix}")

                    yolo_lines = []
                    for ann in img_to_anns.get(img_id, []):
                        cid = ann['category_id']
                        if cid not in cat_id_to_idx:
                            continue
                        cidx = cat_id_to_idx[cid]
                        class_counts[TARGET_CLASSES[cidx]] += 1
                        x, y, bw, bh = ann['bbox']
                        xc = max(0.0, min(1.0, (x + bw / 2.0) / iw))
                        yc = max(0.0, min(1.0, (y + bh / 2.0) / ih))
                        nw = max(0.0, min(1.0, bw / iw))
                        nh = max(0.0, min(1.0, bh / ih))
                        yolo_lines.append(f"{cidx} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

                    dst_lbl = out_lbl_dir / f"{new_stem}.txt"
                    with open(dst_lbl, 'w', encoding='utf-8') as lf:
                        if yolo_lines:
                            lf.write("\n".join(yolo_lines) + "\n")
                continue

            # 2. check for standard yolo format (txt label files)
            img_dir = split_dir / 'images' if (split_dir / 'images').exists() else split_dir
            lbl_dir = split_dir / 'labels' if (split_dir / 'labels').exists() else split_dir

            names_map = {}
            yaml_candidates = list(folder.glob('*.yaml')) + list(folder.glob('*.yml'))
            if yaml_candidates:
                with open(yaml_candidates[0], 'r', encoding='utf-8') as yf:
                    yd = yaml.safe_load(yf)
                    if isinstance(yd.get('names'), list):
                        names_map = {i: n for i, n in enumerate(yd['names'])}
                    elif isinstance(yd.get('names'), dict):
                        names_map = {int(k): v for k, v in yd['names'].items()}

            for src_img in img_dir.iterdir():
                if not src_img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    continue

                new_stem = f"{prefix}_{src_img.stem}"
                shutil.copy(src_img, out_img_dir / f"{new_stem}{src_img.suffix}")

                src_lbl = lbl_dir / f"{src_img.stem}.txt"
                yolo_lines = []
                if src_lbl.exists():
                    with open(src_lbl, 'r', encoding='utf-8') as lf:
                        for line in lf:
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            raw_cid = int(parts[0])
                            raw_name = names_map.get(raw_cid, folder.name)
                            mapped_name = rule(raw_name)
                            if mapped_name in TARGET_CLASSES:
                                new_cid = TARGET_CLASSES.index(mapped_name)
                                class_counts[TARGET_CLASSES[new_cid]] += 1
                                yolo_lines.append(f"{new_cid} {' '.join(parts[1:])}")

                dst_lbl = out_lbl_dir / f"{new_stem}.txt"
                with open(dst_lbl, 'w', encoding='utf-8') as lf:
                    if yolo_lines:
                        lf.write("\n".join(yolo_lines) + "\n")

    # generate data.yaml
    yaml_dict = {
        'path': str(MERGED_DIR.as_posix()),
        'train': 'train/images',
        'val': 'valid/images',
        'names': {i: n for i, n in enumerate(TARGET_CLASSES)}
    }
    with open(MERGED_DIR / 'data.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(yaml_dict, f)

    print(f"[Done] Dataset ready at: {MERGED_DIR}")
    print("  -> Parsed Instances by Class:")
    for cls_name, count in class_counts.items():
        print(f"     * {cls_name}: {count}")

def train_and_export():
    print("[3/4] Starting YOLOv8 Training for 7 classes (v7 with Fire Enhancement)...")
    model = YOLO('yolov8n.pt')

    # hyperparameters optimized for fire detection & windows stability
    results = model.train(
        data=str(MERGED_DIR / 'data.yaml'),
        epochs=100,
        imgsz=640,
        batch=16,
        workers=2,         # prevents memory exhaustion on Windows
        device=0,
        optimizer='AdamW',
        hsv_h=0.02,        # enhanced hue range for fire/flame
        hsv_s=0.75,        # enhanced saturation for fire
        hsv_v=0.45,        # brightness variation
        scale=0.7,         # varied zoom scale for far/near fire
        mosaic=1.0,
        mixup=0.2,         # mixup helps overlapping fire & smoke
        copy_paste=0.3,    # duplicate fire/objects across scenes
        close_mosaic=15,   # disable mosaic for the last 15 epochs for fine tuning
        project=str(WORK_DIR / 'runs_7'),
        name='scout_disaster_v7',
        exist_ok=True
    )

    print("[4/4] Exporting ONNX models...")
    weights_dir = WORK_DIR / 'runs_7' / 'scout_disaster_v7' / 'weights'
    best_weights = weights_dir / 'best.pt'
    trained_model = YOLO(str(best_weights))

    # 1. export 640x640 onnx
    print("  Exporting 640x640 ONNX...")
    exported_640_path = trained_model.export(
        format='onnx',
        imgsz=[640, 640],
        dynamic=False,
        simplify=True,
        opset=12
    )
    final_onnx_640 = weights_dir / 's2m_vAI_lite_640_v7.onnx'
    if Path(exported_640_path).exists():
        shutil.move(exported_640_path, final_onnx_640)

    # 2. export 320x320 onnx
    print("  Exporting 320x320 ONNX...")
    exported_320_path = trained_model.export(
        format='onnx',
        imgsz=[320, 320],
        dynamic=False,
        simplify=True,
        opset=12
    )
    final_onnx_320 = weights_dir / 's2m_vAI_lite_320_v7.onnx'
    if Path(exported_320_path).exists():
        shutil.move(exported_320_path, final_onnx_320)

    # generate labels.txt
    labels_file = WORK_DIR / 's2m_vAI_lite_labels_v7.txt'
    with open(labels_file, 'w', encoding='utf-8') as f:
        for name in TARGET_CLASSES:
            f.write(f"{name}\n")

    print("\n==========================================")
    print("Training and Export Finished (Version 7 - 7 Classes)!")
    print(f"640 ONNX : {final_onnx_640}")
    print(f"320 ONNX : {final_onnx_320}")
    print(f"Labels   : {labels_file}")
    print("==========================================")

if __name__ == '__main__':
    prepare_dataset()
    train_and_export()