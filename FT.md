# NB

## FineTuning the YOLOv8: training on the bvn dataset

chmod +x ft_yolo.sh
./ft_yolo.sh

## Run with args

Test with images

    ```shell
    
        python scripts/yolov8-stereo.py --input data/30.jpg --model baseline
        python scripts/yolov8-stereo.py --input data/30.jpg --model cbam

    ```
    
Test with video

    ```shell
    
        python scripts/yolov8-stereo.py --input data/dual_camera.avi --model baseline
        python scripts/yolov8-stereo.py --input data/dual_camera.avi --model cbam

    ```
    
