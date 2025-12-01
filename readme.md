
QQ group: 957577822 (full)  
QQ group: 546943464 (full)  
QQ group: 258137361


I am a Bilibili Up namded kaylordut, you can find me via [link](https://space.bilibili.com/327258623?spm_id_from=333.1387.0.0)
# Download Model
There is a [repository](https://github.com/kaylorchen/model_convert) with github action to converting model, you can find om(huaiwei), rknn(rockchip) and onnx in the repository.

# Install Dependencies

PS: The project only supports Ubuntu 22.04. If you wish to support other systems, please send an email to kaylor.chen@qq.com. Consultation fees are required. Thank you. 

## Add My software source

- If your device is Jetson(Orin), RK3588 or AiPro
```bash
cat << 'EOF' | sudo tee /etc/apt/sources.list.d/kaylordut.list 
deb [arch=arm64 signed-by=/etc/apt/keyrings/kaylor-keyring.gpg] http://apt.kaylordut.cn/kaylordut/ kaylordut main
EOF
sudo mkdir /etc/apt/keyrings -pv
sudo wget -O /etc/apt/keyrings/kaylor-keyring.gpg http://apt.kaylordut.cn/kaylor-keyring.gpg
```
- if your device is PC
```bash
cat << 'EOF' | sudo tee /etc/apt/sources.list.d/kaylordut.list 
deb [signed-by=/etc/apt/keyrings/kaylor-keyring.gpg] http://apt.kaylordut.cn/kaylordut/ kaylordut main
EOF
sudo mkdir /etc/apt/keyrings -pv
sudo wget -O /etc/apt/keyrings/kaylor-keyring.gpg http://apt.kaylordut.cn/kaylor-keyring.gpg
```

## install software packages
 
### install common packages

```bash
sudo apt update
sudo apt install kaylordut-dev libbytetrack libopencv-dev libyaml-cpp-dev
```
> kaylordut-dev: my private Log library based on spdlog.  
> libbytetrack: ByteTrack library was built by me. you can find it in [repository](https://github.com/kaylorchen/bytetrack)

### install ai-instance
- print ai-instance version
```bash
❯ apt policy ai-instance 
ai-instance:
  Installed: 1.0.0-51-gce55bc3-tensorrt
  Candidate: 1.0.0-51-gce55bc3-tensorrt
  Version table:
 *** 1.0.0-51-gce55bc3-tensorrt 500
        500 http://apt.kaylordut.cn/kaylordut kaylordut/main amd64 Packages
        100 /var/lib/dpkg/status
     1.0.0-51-gce55bc3-onnx 500
        500 http://apt.kaylordut.cn/kaylordut kaylordut/main amd64 Packages
```
- select ai-instance version
```bash
apt install -y ai-instance=1.0.0-51-gce55bc3-tensorrt # if your device is jetson or RTX device, select tensorrt version
# OR
apt install -y ai-instance=1.0.0-51-gce55bc3-onnx # if your device is PC without GPU, select onnx version
# OR
apt install -y ai-instance=xxxxx-rknn # if your device is Rockchip, select rknn version
# OR
apt install -y ai-instance=xxxxx-nnrt # if your device is AI Pro, select nnrt version
```
> please install libnvinfer-plugin10 and libnvinfer10 before installing ai-instance(tensorrt version)  
> Jetson(Orin) device: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb  
> RTX device: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb  


# TEST

```bash
mkdir -pv build
cd build
cmake ..
```
> Please modify your configuration in the config directory, such as the model path, camera index, etc.

## Test Yolo
Enter the bulild directory, and run the following command
```bash
make yolo_mutilthreading_demo -j$(nproc) # compile yolo demo only
./yolo_mutilthreading_demo
```

## Test Depth Anything
```bash
./depth_demo 
```



