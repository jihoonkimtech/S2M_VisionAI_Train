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
MERGED_DIR = WORK_DIR / "merged_dataset"
EXTRACT_DIR = WORK_DIR / "datasets"

TARGET_CLASSES = [
    'person',
    'fire',
    'smoke',
    'exit_indicator',
    'gas_tank',
    'fire_extinguisher'
]

ZIP_FILES = [
    'Yolo-disaster-relief.zip',
    'exit-sign-Extended.zip',
    'fire-and-person-detection.zip',
    'fire-smoke-detection.zip',
    'gas-tank.zip',
    'fire-extinguisher.zip'
]

DATASET_RULES = {
    'fire-extinguisher': lambda name: 'fire_extinguisher',
    'gas-tank': lambda name: 'gas_tank',
    'exit-sign-Extended': lambda name: 'exit_indicator',
    'Yolo-disaster-relief': lambda name: 'person',
    'fire-smoke-detection': lambda name: 'smoke' if 'smoke' in name.lower() else 'fire',
    'fire-and-person-detection': lambda name: (
        'person' if any(k in name.lower() for k in ['person', 'human', 'victim'])
        else ('smoke' if 'smoke' in name.lower() else 'fire')
    ),
}
# =================================================

def prepare_dataset():
    print("[1/4] Extracting zip files...")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    for zname in ZIP_FILES:
        zpath = SRC_ZIP_DIR / zname
        if not zpath.exists():
            print(f"  [Skip] Zip not found: {zpath}")
            continue
        out_dir = EXTRACT_DIR / zpath.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath, 'r') as zf:
            zf.extractall(out_dir)

    print("[2/4] Parsing and converting COCO JSON to YOLO format...")
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)

    for split in ['train', 'valid']:
        (MERGED_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

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

            json_files = list(split_dir.glob('*.json'))
            if not json_files:
                continue

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

            out_img_dir = MERGED_DIR / tgt_split / 'images'
            out_lbl_dir = MERGED_DIR / tgt_split / 'labels'

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

def train_and_export():
    print("[3/4] Starting YOLOv8 Training...")
    model = YOLO('yolov8n.pt')

    results = model.train(
        data=str(MERGED_DIR / 'data.yaml'),
        epochs=100,
        imgsz=640,
        batch=16,          # adjust batch size to 8 or 32 based on GPU VRAM
        device=0,          # GPU index
        optimizer='AdamW',
        hsv_v=0.4,
        mosaic=1.0,
        project=str(WORK_DIR / 'runs'),
        name='scout_disaster_multi',
        exist_ok=True
    )

    print("[4/4] Exporting ONNX models...")
    weights_dir = WORK_DIR / 'runs' / 'scout_disaster_multi' / 'weights'
    best_weights = weights_dir / 'best.pt'
    trained_model = YOLO(str(best_weights))

    # 1. export 640x640 onnx and rename immediately to prevent overwriting
    print("  Exporting 640x640 ONNX...")
    exported_640_path = trained_model.export(
        format='onnx',
        imgsz=[640, 640],
        dynamic=False,
        simplify=True,
        opset=12
    )
    final_onnx_640 = weights_dir / 's2m_vAI_lite_640.onnx'
    if Path(exported_640_path).exists():
        shutil.move(exported_640_path, final_onnx_640)

    # 2. export 320x320 onnx and rename
    print("  Exporting 320x320 ONNX...")
    exported_320_path = trained_model.export(
        format='onnx',
        imgsz=[320, 320],
        dynamic=False,
        simplify=True,
        opset=12
    )
    final_onnx_320 = weights_dir / 's2m_vAI_lite_320.onnx'
    if Path(exported_320_path).exists():
        shutil.move(exported_320_path, final_onnx_320)

    # generate labels.txt
    labels_file = WORK_DIR / 's2m_vAI_lite_labels.txt'
    with open(labels_file, 'w', encoding='utf-8') as f:
        for name in TARGET_CLASSES:
            f.write(f"{name}\n")

    print("\n==========================================")
    print("Training and Export Finished!")
    print(f"640 ONNX : {final_onnx_640}")
    print(f"320 ONNX : {final_onnx_320}")
    print(f"Labels   : {labels_file}")
    print("==========================================")

if __name__ == '__main__':
    prepare_dataset()
    train_and_export()