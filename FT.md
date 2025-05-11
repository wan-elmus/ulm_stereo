# NB

## FineTuning the YOLOv8: training on the bvn dataset

chmod +x ft_yolo.sh
./ft_yolo.sh

## Run with args

Test with images

    ```shell
    
        python3 scripts/yolov8-stereo.py --input data/30.jpg --model baseline3
        python3 scripts/yolov8-stereo.py --input data/30.jpg --model superior4

    ```
    
Test with video

    ```shell
    
        python3 scripts/yolov8-stereo.py --input data/dual_camera.avi --model baseline3
        python3 scripts/yolov8-stereo.py --input data/dual_camera.avi --model superior4

    ```
    
