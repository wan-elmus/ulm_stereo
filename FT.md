# NB

## FineTuning the YOLOv8: training on the bvn dataset

chmod +x ft_yolo.sh
./ft_yolo.sh

## Run with args

Test with images

    ```shell
    
        python3 scripts/yolov8-stereo.py --input data/30.jpg --model baseline42
        python3 scripts/yolov8-stereo.py --input data/30.jpg --model superior5

    ```
    
Test with video

    ```shell
    
        python3 scripts/yolov8-stereo.py --input data/dual_camera_20250505_163126.avi --model baseline42
        python3 scripts/yolov8-stereo.py --input data/dual_camera_20250505_163126.avi --model superior5

    ```
    