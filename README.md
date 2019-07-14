# RasPiCam Vision Lite

RasPiCam Vision Lite is a minimalistic and lightweight [Flask](https://palletsprojects.com/p/flask) web app that serves on [Raspberry Pi](https://www.raspberrypi.org) and streams live video from its camera module at high framerates while doing on-device image classification asynchronously with [TensorFlow Lite](https://www.tensorflow.org/lite) models. 

With RasPiCam Vision Lite, efficiently serving and comparing multiple TensorFlow Lite models for image classification is just a few clicks away. It takes advantage of multiprocessing and shifts the computational heavy lifting of inferencing into dedicated subprocesses, independent of video streaming feed. 

![demo](img/demo.png)

## :strawberry:Overview



## :strawberry:Usage

1. Open a terminal, SSH into your Raspberry Pi and clone the repository.

2. *(Optional)* Put your custom trained TensorFlow Lite [quantized](https://www.tensorflow.org/lite/performance/post_training_quantization) model `{your-model-version}.tflite` and labels `labels.txt` into a same separate folder `{your-model-version}/` under `models/` directory. Of course, you can still download and use some other offically released [hosted models](https://www.tensorflow.org/lite/guide/hosted_models#quantized_models) as this project does.

```
.
└─ models
   └── {your-model-version}
       ├── {your-model-version}.tflite
       └── labels.txt
```


3. Start a Flask web server by ```python3 main.py``` under repository root.

4. Open a browser and go to the IP address with port 5000 (e.g. `192.168.0.104:5000`) distributed to Raspberry Pi in your local network. You'll be greeted by the web interface as shown above.

5. Go select one of the models from the dropdown list, and press `SERVE` to watch video live streaming as the serving model does inferencing in the background. Try selecting a different model and tap `SERVE` again. 

6. Shut down the server safely by clicking `SHUTDOWN`.

## :strawberry:How It Works



## :strawberry:Dependencies

Following configurations are fully tested. Some variations could also work, but no guarantee.

**Hardwares**
- Raspberry Pi 3 Model B+
- Pi Camera Module V2

**Softwares**
- Raspbian 9 (stretch)
- Python 3.5+
- TensorFlow 2.0 
- OpenCV 4.1
- Flask 1.1.0
- Picamera 1.13
- Chrome 75, Safari 12

## :strawberry:License

[MIT](https://github.com/jingw222/raspicam_vision_lite/blob/master/LICENSE)