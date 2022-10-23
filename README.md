# Long-tail Detection with Effective Class-Margins 

## Introduction 

This is an official implementation of [**Long-tail Detection with Effective Class-Margins**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680684.pdf). 

> [**Long-tail Detection with Effective Class-Margins**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680684.pdf)             
> [Jang Hyun Cho](https://janghyuncho.github.io/) and [Philipp Kr&auml;henb&uuml;hl](https://www.philkr.net/)                 
> *[ECCV 2022](https://eccv2022.ecva.net/) (oral)*      

Contact: janghyuncho7 [at] utexas.edu. 

## Installation
### Requirements 
- Python 3.6+
- PyTorch 1.8+
- torchvision 0.9+
- mmdet 2.14+
- mmcv 1.3+

We tested our codebase on mmdet 2.24.1, mmcv 1.5.1, PyTorch 1.11.0, torchvision 0.12.0, and python 3.9. 

### Setup
To setup the code, follow the commands below:

~~~
# Clone the repo.
git clone git@github.com:janghyuncho/ECM-Loss.git
cd ECM-Loss 

# Create conda env.
conda create --name ecm_loss python=3.8 -y 
conda activate ecm_loss

# Install PyTorch.
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

# Install mmcv.
pip install -U openmim
mim install mmcv-full

# And mmdetection. 
pip install mmdet 

# Install mmdet dependencies.
pip install -e .

# Additionally, install lvis-api. 
pip install lvis
~~~

### Dataset 
Please download [LVIS dataset](https://www.lvisdataset.org/dataset), and structure the folders as following. 
~~~
data
  ├── lvis_v1
  |   ├── annotations
  │   │   │   ├── lvis_v1_val.json
  │   │   │   ├── lvis_v1_train.json
  │   ├── train2017
  │   │   ├── 000000004134.png
  │   │   ├── 000000031817.png
  │   │   ├── ......
  │   ├── val2017
  │   ├── test2017
~~~

## Training with ECM-Loss 
All training commands for our models can be found [here](https://github.com/janghyuncho/ECM-Loss/tree/main/sh_files/ecm_loss). For example, you can train *mask-rcnn* with *resnet-50* backbone for 12 epochs with the following command:
~~~
./sh_files/ecm_loss/r50_1x.sh 
~~~

ECM Loss is implemented [here](https://github.com/janghyuncho/ECM-Loss/blob/main/mmdet/models/losses/effective_class_margin_loss.py).

## Pretrained Models on LVIS v1


| Framework | Backbone | Schedule | Box AP | Mask AP | Weight | Config |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|Mask R-CNN |R50    | 1x  |26.9 | 26.4|[weight]()|[config](https://github.com/janghyuncho/ECM-Loss/blob/main/configs/effective_class_margin_loss/r50_ecm_1x.py)  |
|Mask R-CNN |R50    | 2x  |27.9 | 27.5|[weight]()|[config](https://github.com/janghyuncho/ECM-Loss/blob/main/configs/effective_class_margin_loss/r50_ecm_2x.py)  |
|Mask R-CNN |R101   | 2x  |29.4 | 28.7|[weight]()|[config](https://github.com/janghyuncho/ECM-Loss/blob/main/configs/effective_class_margin_loss/r101_ecm_2x.py)  |
|Cascade Mask R-CNN |R101 | 2x | 33.4 | 30.6 |[weight]()|[config](https://github.com/janghyuncho/ECM-Loss/blob/main/configs/effective_class_margin_loss/c101_ecm_2x.py)  |

## Citation
If you use use ECM Loss, please cite our paper:

	@inproceedings{cho2022ecm,
  		title={Long-tail Detection with Effective Class-Margins},
  		author={Jang Hyun Cho and Philipp Kr{\"a}henb{\"u}hl},
  		booktitle={European Conference on Computer Vision (ECCV)},
  		year={2022}
	}


## Acknowledgement 
ECM Loss is based on [MMDetection](https://github.com/open-mmlab/mmdetection). 
