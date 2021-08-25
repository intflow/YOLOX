



## Introduction
yolox deepstream을 실행하기 위한 repo이다.
vscode debugmode를 사용하기위해 python의 argument의 경로들이 절대경로 이므로 경로 오류시 본인 경로에 맞게 설정하고 run 하길 바란다.


deepstream을 위해 
1. step] torch model을 onnx로 변환하고 [torch2onnx]

2. step] onnx모델읠 tensorRT engine으로 변환후 [onnx2trt]

3. step] yolox-deepstream에서 deepstream을 실행하여야한다.[deepstream]



## Quick Start




<details>
<summary>torch2onnx</summary>

# Convert torch Model to ONNX
First, you should move to <YOLOX_HOME> by:
```shell
pip install loguru

cd <YOLOX_HOME>
```
Then, you can:

1. Convert a standard YOLOX model by -n:
```shell
python3 tools/export_onnx.py --output-name yolox_s.onnx -n yolox-s -c yolox_s.pth
```
Notes:
* -n: specify a model name. The model name must be one of the [yolox-s,m,l,x and yolox-nane, yolox-tiny, yolov3]
* -c: the model you have trained
* -o: opset version, default 11. **However, if you will further convert your onnx model to [OpenVINO](https://github.com/Megvii-BaseDetection/YOLOX/demo/OpenVINO/), please specify the opset version to 10.**
* --no-onnxsim: disable onnxsim
* To customize an input shape for onnx model,  modify the following code in tools/export.py:

    ```python
    dummy_input = torch.randn(1, 3, exp.test_size[0], exp.test_size[1])
    ```

2. Convert a standard YOLOX model by -f. When using -f, the above command is equivalent to:

```shell
python3 tools/export_onnx.py --output-name yolox_s.onnx -f exps/default/yolox_s.py -c yolox_s.pth
```

3. To convert your customized model, please use -f:

```shell
python3 tools/export_onnx.py --output-name your_yolox.onnx -f exps/your_dir/your_yolox.py -c your_yolox.pth
```



# onnx 추론


Step1.
```shell
cd demo/ONNXRuntime
```

Step2. 
```shell
python3 onnx_inference.py -m <ONNX_MODEL_PATH> -i <IMAGE_PATH> -o <OUTPUT_DIR> -s 0.3 --input_shape 640,640
```
Notes:
* -m: your converted onnx model
* -i: input_image
* -s: score threshold for visualization.
* --input_shape: should be consistent with the shape you used for onnx convertion.

</details>


<details>
<summary>onnx2trt</summary>

# onnx2trt
tools/trt.py를 통하여 trt변환을 할 수 있지만(torch2trt) 

[onnx2trt](https://github.com/onnx/onnx-tensorrt/tree/master) 를 사용하였다.

```
git clone https://github.com/onnx/onnx-tensorrt.git
cd onnx-tensorrt
mkdir build && cd build
cmake .. -DTENSORRT_ROOT=<path_to_trt> && make -j
// Ensure that you update your LD_LIBRARY_PATH to pick up the location of the newly built library:
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
```

# Convert onnx to trt

```
onnx2trt yolox_d54_sim.onnx -o yolox_d54_fp16.engine -b 1 -d 16
```
Notes:
* [-o engine_file.trt]  (output TensorRT engine)" << "\n"
* [-m onnx_model_out.pb] (output ONNX model)" << "\n"
* [-b max_batch_size (default 32)]" << "\n"
* [-w max_workspace_size_bytes (default 1 GiB)]" << "\n"
* [-d model_data_type_bit_depth] (32 => float32, 16 => float16)" << "\n"
* [-O passes] (optimize onnx model. Argument is a semicolon-separated list of passes)" << "\n"


# trt 추론

```shell
cd YOLOX/demo/TensorRT/cpp
mkdir build
cd build
cmake ..
make
```

Then run the demo:

```shell
./yolox ../model_trt.engine -i ../../../../assets/dog.jpg
```

</details>


<details open>
<summary>deepstream</summary>

# deepstream

cd YOLO-deepstream/nvdsinfer_custom_impl_yolox/

make
```

use to parse infer postprocess.

# Run

```shell

deepstream-app -c deepstream_app_config.txt
```

</details>





<img src="assets/det_res.jpg" width="300" >