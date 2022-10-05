*코드 출처
https://github.com/shaojiawei07/VL-VFE.git

코드 수정 목록
```python
# 자동으로 mnist 설치되도록 코드 수정했음. - 22.09.20 -
# save 파일명에 : 있어서 오류나는거 _로 수정했음 -22.09.25 -
# nn.functional.sigmoid에서 torch.sigmoid로 수정함 - 22.09.26 -
# MNIST test code 완료 - 22.09.26 -
# CIFAR test code 완료 - 22.09.26 -
# torchviz로 모델 png파일로 저장할 수 있음. - 22.09.27 -
# 모델 시각화 코드 추가함.  - 22.09.28 - 
```
# VL-VFE
This repository contains the codes for the variable-length variational feature encoding (VL-VFE) method proposed in the [paper](https://arxiv.org/pdf/2102.04170.pdf) "Learning Task-Oriented Communication for Edge Inference: An Information Bottleneck Method", which is accepted to IEEE Journal on Selected Areas in Communications.

## Dependencies
### Packages
```
Pytorch 1.8.1
Torchvision 
cuda 10.2 or cuda 11.3 or cpu only
```
cuda 버전 확인 
```sh
nvcc --version
```

### Datasets
```
MNIST(자동으로 설치되도록 코드 수정했음.)
CIFAR-10(이 데이터가 없다면 코드에서 download를 검색해서 True로 고쳐준 후 한번 돌리면 다운됨, True로 해놓으면 계속 경고문이 떠서 False로 함)
```

### How to install
1. python version을 지정을 안하고 생성해야 설치가 되네요(3.8.13으로 설치됨)(아마도 conda env의 python version을 지정하면서 생기는 package들이 충돌을 일으킴)
```sh
conda create -n vl-vfe
```
2. activate
```sh
activate vl-vfe
cd PycharmProjects\deepsc\VL-VFE
```

3. install requirements
```sh
# CUDA 10.2
conda install pytorch==1.8.1 torchvision==0.9.1  cudatoolkit=10.2 -c pytorch

# CUDA 11.3
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
# conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 -c pytorch -c conda-forge
위와 같이 설치하면 cudatoolkit 11.1로 자동설치됨.
cudatoolkit=11.3 하면 cpu_only 설치되고, 파이토치도 cpu_0 버전으로 바껴서 gpu 연동 안되는듯?
근데 이렇게 하면 파이썬은 3.6.15로 설치됨.

# CPU Only
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cpuonly -c pytorch
``` 
4. 모델 시각화 코드를 위한 torchviz 설치
```sh
pip install torchviz
```

## How to run


### default parameter(mnist)
```python 
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--channel_noise', type=float, default=0.3162)
parser.add_argument('--intermediate_dim', type=int, default=64)
parser.add_argument('--beta', type=float, default=1e-3)
parser.add_argument('--threshold', type=float, default=1e-2)
parser.add_argument('--test', type=int, default=0)
parser.add_argument('--weights', type=str)
parser.add_argument('--model_viz', type=bool, default = 0)
```


### Train the VL-VFE method on the MNIST dataset
```sh
python VL_VFE_MNIST.py --intermediate_dim 64  --beta 6e-3 --threshold 1e-2 --model_viz 1
```

#### Hyeongeon Train
```shell
python VL_VFE_MNIST.py --intermediate_dim 64  --beta 6e-3 --threshold 1e-2 --epochs 1
```
### default parameter(mnist)
```python 
parser.add_argument('--intermediate_dim', type=int, default=64)
parser.add_argument('--epochs', type=int, default=320, help='epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=1e-2)
parser.add_argument('--threshold', type=float, default=1e-2)
parser.add_argument('--decay_step', type=int, default=60)
parser.add_argument('--test', type=int, default=0)
parser.add_argument('--weights', type=str)
parser.add_argument('--channel_noise', type=float, default = 0.1)
parser.add_argument('--model_viz', type=bool, default = 0)
```

### Train the VL-VFE method on the CIFAR dataset
```sh
python VL_VFE_CIFAR.py --intermediate_dim 64  --beta 9e-3 --threshold 1e-2
```

#### Hyeongeon Train
```shell
python VL_VFE_CIFAR.py --intermediate_dim 64  --beta 6e-3 --threshold 1e-2 --epochs 1 --model_viz 1 
```

The parameter `intermediate_dim` denotes the (maximum) dimension of the encoded feature vector. The weighting factor `beta` and the pruning threshold `threshold` control the tradeoff between the accuracy and the number of activated dimensions.



## Inference
After training the neural network, we can test the performance under different channel conditions `--channel_noise`, which represents the standard deviation in the Gaussian distribution. The relationship between the `--channel_noise` and the peak signal-to-noise ratio (PSNR) is summarized as follows:

| `channel_noise` | 0.3162 |0.2371|0.1778|0.1334|0.1000|0.0750|0.0562|
| :---: | :---: | :---: | :---: |:---: | :---: |:---: | :---: |
|PSNR|10 dB|12.5 dB|15 dB|17.5 dB| 20 dB| 22.5 dB| 25 dB|



### Test the VL-VFE method on the MNIST dataset with PSNR=20 dB
```sh
python VL_VFE_MNIST.py --test 1 --intermediate_dim 64 --channel_noise 0.1 --threshold 1e-2 --weights ./pretrained/model/location
```

#### Hyeongeon Test( 마지막에 예시처럼 파일명을 집어넣어야함.)
```sh
python VL_VFE_MNIST.py --test 1 --intermediate_dim 64 --channel_noise 0.1 --threshold 1e-2 --model_viz 1 --weights MNIST_model_dim_64_beta_0.006_accuracy_94.1880_model.pth
```

### Test the VL-VFE method on the CIFAR dataset with PSNR=20 dB
```sh
python VL_VFE_CIFAR.py --test 1 --intermediate_dim 64 --channel_noise 0.1 --threshold 1e-2 --weights ./pretrained/model/location
```

#### Hyeongeon Test( 마지막에 예시처럼 파일명을 집어넣어야함.)
```sh
python VL_VFE_CIFAR.py --test 1 --intermediate_dim 64 --channel_noise 0.1 --threshold 1e-2 --weights CIFAR_model_dim_30_beta_0.006_accuracy_92.6_model.pth
```

Several pretrained models and results are shown in [Examples](https://github.com/shaojiawei07/VL-VFE/tree/main/Examples).


## Citation

```
@article{shao2021learning,  
  author={Shao, Jiawei and Mao, Yuyi and Zhang, Jun},  
  journal={IEEE Journal on Selected Areas in Communications},  
  title={Learning Task-Oriented Communication for Edge Inference: An Information Bottleneck Approach},   
  year={2022},  
  volume={40},  
  number={1},  
  pages={197-211},  
  doi={10.1109/JSAC.2021.3126087}}
```
## Others

* The variational feature encoding (VFE) proposed in this paper can be achieved by replacing the function `self.gamma_mu = gamma_function()` with a vector `self.mu = nn.Parameter(torch.ones(args.intermediate_dim))` and fixing the channel noise level in the training process.


* Known problem: The loss may become `NaN` when training the network on the CIFAR dataset.

