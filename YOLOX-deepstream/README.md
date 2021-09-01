

# YOLOX(Megvii-BaseDetection) DeepStream 



# News

Deploy yolox to deep stream, FPS > 70 - 2021-7-21

# System Requirements

cuda 10.0+

TensorRT 7+

OpenCV 4.0+ (build with opencv-contrib module) [how to build](https://gist.github.com/nanmi/c5cc1753ed98d7e3482031fc379a3f3d#%E6%BA%90%E7%A0%81%E7%BC%96%E8%AF%91gpu%E7%89%88opencv)

OpenMP

DeepStream 5.0+



# Installation

```bash


cd nvdsinfer_custom_impl_yolox/

make
```

use to parse infer postprocess.

# Run

```shell

deepstream-app -c deepstream_app_config.txt
```

