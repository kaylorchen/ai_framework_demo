
QQ group: 957577822 (full)  
QQ group: 546943464 (full)  
QQ group: 258137361


I am a Bilibili Up namded kaylordut, you can find me via [link](https://space.bilibili.com/327258623?spm_id_from=333.1387.0.0)
# Download Model
There is a [repository](https://github.com/kaylorchen/model_convert) with github action to converting model, you can find om(huaiwei), rknn(rockchip) and onnx in the repository.

# Install Dependencies

PS: The project only supports Ubuntu 22.04. If you wish to support other systems, please send an email to kaylor.chen@qq.com. Consultation fees are required. Thank you. 

## add my software source

- If your device is RK3588 or AiPro
```bash
cat << 'EOF' | sudo tee /etc/apt/sources.list.d/kaylordut.list 
deb [arch=arm64 signed-by=/etc/apt/keyrings/kaylor-keyring.gpg] http://apt.kaylordut.cn/kaylordut/ kaylordut main
deb [arch=arm64 signed-by=/etc/apt/keyrings/kaylor-keyring.gpg] http://apt.kaylordut.cn/rk3588/ubuntu jammy main
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

```bash
sudo apt update
sudo apt install kaylordut-dev libbytetrack libopencv-dev libyaml-cpp-dev
```

- check ai-instance version
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

```bash
apt install -y ai-instance=1.0.0-51-gce55bc3-tensorrt # tensorrt
# OR
apt install -y ai-instance=1.0.0-51-gce55bc3-onnx # onnxruntime
```
> please install libnvinfer-plugin10 and libnvinfer10 before installing ai-instance(tensorrt version)


# TEST

```bash
mkdir -pv build
cd build
cmake ..
make
```
> Please modify your configuration in the config directory

## Test Yolo
Enter the bulild directory, and run the following command
```bash
./yolo_mutilthreading_demo
```

## Test Depth Anything
```bash
./depth_demo 
```



