import shutil
from pathlib import Path
from ultralytics import YOLO

# locate trained weights
weights_dir = Path(r"C:\Users\kimwl\OneDrive\바탕 화면\train_S2M\workspace\runs\scout_disaster_multi\weights")
best_pt = weights_dir / "best.pt"

model = YOLO(str(best_pt))

# 1. export 640x640 ONNX and rename
print("Exporting 640x640 ONNX...")
exported_640 = model.export(
    format='onnx',
    imgsz=[640, 640],
    dynamic=False,
    simplify=True,
    opset=12
)
target_640 = weights_dir / "s2m_vAI_lite_640.onnx"
shutil.move(exported_640, target_640)

# 2. export 320x320 ONNX and rename
print("Exporting 320x320 ONNX...")
exported_320 = model.export(
    format='onnx',
    imgsz=[320, 320],
    dynamic=False,
    simplify=True,
    opset=12
)
target_320 = weights_dir / "s2m_vAI_lite_320.onnx"
shutil.move(exported_320, target_320)

print("\n==========================================")
print(f"640 ONNX Saved: {target_640}")
print(f"320 ONNX Saved: {target_320}")
print("==========================================")