# AutoShape
ICCV2021 Paper: AutoShape: Real-Time Shape-Aware Monocular 3D Object Detection

[arXiv](https://arxiv.org/abs/2108.11127)
## Auto-labeled Car Shape for KITTI
We release our Auto-labeled car shape data for KITTI with COCO formate. Each car instance has been assigned a 3D model. [Trainset](https://drive.google.com/file/d/1U6d4Z0l4FsAKUiv6jehT7esgsJ5ULWaI/view?usp=sharing) and [Valset](https://drive.google.com/file/d/1KfHiPOjWyV-pW3jxyTogzG07KjLvsF2g/view?usp=sharing) with  3000 vertexes 3D models annotations can be downloaded from Google Drive. 


<img src="https://github.com/zongdai/AutoShape/blob/main/README/autoshape_data_exmaple.png" width="860"/>

### Data Formate
```python
# we add 2D/3D keypoints in KITTI car instance annotations
annotations: [
    '2dkeypoints': list # (3000 + 9) * 3 (u, v, visiblity),
    '3dkeypoints': list # (3000 + 9) * 3 (x, y, z in model local coordinate)
    ...
    ], ...
```
## Paddle Implement(incomplete)
### Requirements
*   Ubuntu 18.04
*   Python 3.7
*   PaddlePaddle 2.1.0
*   CUDA 10.2
### PaddlePaddle installation
```bash

conda create -n paddle_latest python=3.7

conda actviate paddle_latest

pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

pip install -r requirement.txt
```
## Pytorch Implement
### Install
1. Install pytorch1.0.0:
    ~~~
    conda install pytorch=1.0.0 torchvision -c pytorch
    ~~~
2. Install the requirements
    ~~~
    pip install -r requirements.txt
    ~~~
3. Compile deformable convolutional (from [DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_0.4)).
    ~~~
    cd $AutoShape_ROOT/pytorch/src/lib/models/networks/ 
    unzip DCNv2-pytorch_1.0.zip
    cd DCNv2
    ./make.sh
    ~~~
4. Compile iou3d (from [pointRCNN](https://github.com/sshaoshuai/PointRCNN)). GCC>4.9, I have tested it with GCC 5.4.0 and GCC 4.9.4, both of them are ok. 
    ~~~
    cd $AutoShape_ROOT/pytorch/src/lib/utiles/iou3d
    python setup.py install
    ~~~
### Dataset preparation
Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and [AutoShape keypoints annotations](https://drive.google.com/file/d/1iMKU9OGLbNRHqclQUd9O9JUrQ0qNkgjy/view?usp=sharing) organize the downloaded files as follows: 
```
pytorch
├── kitti_format
│   ├── data
│   │   ├── kitti
│   │   |   ├── annotations_48 / kitti_train.json .....
│   │   |   ├── annotations_16 / kitti_train.json .....
│   │   │   ├── calib /000000.txt .....
│   │   │   ├── image(left[0-7480] right[7481-14961] for data augmentation)
│   │   │   ├── label /000000.txt .....
|   |   |   ├── train.txt val.txt trainval.txt
├── src
├── requirements.txt
``` 
### Training
Run following command to train model with DLA-34 backbone and 57(48+9) keypoints with 2 GPUs.
   ~~~
   cd pytorch
   python ./src/main.py --data_dir ./kitti_format --exp_id AutoShape_dla34_trainval_rightaug --arch dla_34 --num_joints 57 --sample_pc 48 --batch_size 16 --master_batch_size 8 --lr 1.5e-4 --gpus 0,1 --num_epochs 200 --stereo_aug
   ~~~

### Inference
~~~
python ./src/faster.py --demo  test_image_dir_path --calib_dir calib_dir_path --load_model trained_model_path --gpus 0 --arch dla_34 --num_joints 57 --sample_pc 48
~~~
### Kitti TestServer Evaluation Model

- Training on KITTI trainval split and evaluation on test server.
    - Backbone: DLA-34
    - Num Keypoints: 48 + 9
    - Model: ([Google Drive](https://drive.google.com/file/d/1mTIl2pSw1ekL4i7BmmO_HGtCjS_hFCTf/view?usp=sharing))
    
| Class      |Easy      | Moderate     |Hard       |
| :----:     | :----:   | :----:       |:----:                   
| Car        | 22.47    | 14.17        | 11.36    


## Acknowledgement
- [**RTM3D**](https://github.com/Banconxuan/RTM3D)
- [**CenterNet**](https://github.com/xingyizhou/CenterNet)
## License

AutoShape is released under the MIT License (refer to the LICENSE file for details).
Some of the code are borrowed from, [RTM3D](https://github.com/Banconxuan/RTM3D), [CenterNet](https://github.com/xingyizhou/CenterNet), [dla](https://github.com/ucbdrive/dla) (DLA network), [DCNv2](https://github.com/CharlesShang/DCNv2)(deformable convolutions), [iou3d](https://github.com/sshaoshuai/PointRCNN) and [kitti_eval](https://github.com/prclibo/kitti_eval) (KITTI dataset evaluation). Please refer to the original License of these projects.
## Citation

If you find this project useful for your research, please use the following BibTeX entry.

    
    @inproceedings{liu2021autoshape,
      title={AutoShape: Real-Time Shape-Aware Monocular 3D Object Detection},
      author={Liu, Zongdai and Zhou, Dingfu and Lu, Feixiang and Fang, Jin and Zhang, Liangjun},
      booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
      pages={15641--15650},
      year={2021}
    }
    