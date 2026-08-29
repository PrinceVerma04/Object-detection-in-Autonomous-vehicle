"""
Fine-tunes a YOLO11n checkpoint (COCO-pretrained, already knows 'cow') on the unified
7-class civic-services dataset. YOLO11n is chosen for edge/mobile deployment: ~5-6MB,
one-command export to TFLite/CoreML/ONNX/NCNN, fastest CPU inference in the YOLO family.

Usage:
    python train.py --epochs 100 --imgsz 640
"""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs",
        name="civic_services",
        patience=20,
        mosaic=1.0,
        degrees=5.0,
        fliplr=0.5,
    )


if __name__ == "__main__":
    main()
