#!/bin/bash

# # Train Model 1 (Baseline)
# yolo train \
#     model=yolov8n.pt \
#     data=configs/yolo-bvn.yaml \
#     epochs=100 \
#     imgsz=640 \
#     batch=16 \
#     lr0=0.001 \
#     patience=30 \
#     augment=True \
#     hsv_h=0.015 \
#     hsv_s=0.7 \
#     hsv_v=0.4 \
#     degrees=10.0 \
#     translate=0.1 \
#     scale=0.5 \
#     shear=2.0 \
#     mosaic=1.0 \
#     project=runs/detect \
#     name=train_baseline \
#     device=cpu \
#     save=True \
#     plots=True

# # Train Model 2 (Improved with CBAM)
# python - <<EOF
# from scripts.cbam import CBAM
# from ultralytics import YOLO

# model = YOLO("configs/yolov8-cbam.yaml")
# model.train(
#     data="configs/yolo-bvn.yaml",
#     epochs=100,
#     imgsz=640,
#     batch=16,
#     lr0=0.001,
#     patience=30,
#     augment=True,
#     hsv_h=0.015,
#     hsv_s=0.7,
#     hsv_v=0.4,
#     degrees=10.0,
#     translate=0.1,
#     scale=0.5,
#     shear=2.0,
#     mosaic=1.0,
#     project="runs/detect",
#     name="train_cbam",
#     device="cpu",
#     save=True,
#     plots=True
# )
# EOF

# # Train Model 2 (Improved with SE)
# python - <<EOF
# try:
#     from ultralytics.nn.modules import SE
# except ImportError:
#     from scripts.se import SE
# from ultralytics import YOLO

# model = YOLO("configs/yolov8-se.yaml")
# model.train(
#     data="configs/yolo-bvn.yaml",
#     epochs=100,
#     imgsz=640,
#     batch=16,
#     lr0=0.001,
#     patience=30,
#     augment=True,
#     hsv_h=0.015,
#     hsv_s=0.7,
#     hsv_v=0.4,
#     degrees=10.0,
#     translate=0.1,
#     scale=0.5,
#     shear=2.0,
#     mosaic=1.0,
#     project="runs/detect",
#     name="train_se",
#     device="cpu",
#     save=True,
#     plots=True,
#     loss_cls_weight=[1.0, 1.0, 2.0]
# )
# EOF

# Train Model 2 (Enhanced neck, pretrained on baseline)
yolo train \
    model=configs/yolov8-custom.yaml \
    pretrained=runs/detect/train_baseline3/weights/best.pt \
    data=configs/yolo-bvn.yaml \
    epochs=100 \
    imgsz=640 \
    batch=8 \
    lr0=0.0001 \
    optimizer=SGD \
    momentum=0.937 \
    patience=50 \
    dropout=0.1 \
    freeze=5 \
    strict=False \
    augment=True \
    hsv_h=0.015 \
    hsv_s=0.7 \
    hsv_v=0.4 \
    degrees=10.0 \
    translate=0.1 \
    scale=0.5 \
    shear=2.0 \
    mosaic=1.0 \
    mixup=0.2 \
    project=runs/detect \
    name=train_custom4 \
    device=cpu \
    save=True \
    plots=True