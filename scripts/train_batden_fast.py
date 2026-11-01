#!/usr/bin/env python3
import os
import torch
from ultralytics import YOLO

def main():
    data_yaml = os.path.abspath("datasets/batden_fast/data.yaml")
    output_model_dir = "models"
    os.makedirs(output_model_dir, exist_ok=True)

    print("Initializing YOLO11n-seg for ultra-fast agricultural path segmentation...")
    model = YOLO("yolo11n-seg.pt")

    # Set threads for training efficiency
    torch.set_num_threads(min(8, os.cpu_count() or 4))

    print("Starting training on CPU with imgsz=256, batch=16, epochs=30...")
    results = model.train(
        data=data_yaml,
        epochs=30,
        imgsz=256,
        batch=16,
        device="cpu",
        workers=2,
        project="runs_train",
        name="batden_fast_256",
        exist_ok=True,
        save=True,
        verbose=True,
        plots=False,
        val=True
    )

    best_pt = os.path.join("runs_train", "batden_fast_256", "weights", "best.pt")
    target_pt = os.path.join(output_model_dir, "batden_fast_256.pt")

    if os.path.exists(best_pt):
        import shutil
        shutil.copy(best_pt, target_pt)
        print(f"SUCCESS: Model saved to {target_pt}")
    else:
        print(f"Error: {best_pt} not found")

if __name__ == "__main__":
    main()

