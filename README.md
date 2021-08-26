# AutoShape
ICCV2021 Paper: AutoShape: Real-Time Shape-Aware Monocular 3D Object Detection

[arXiv](https://arxiv.org/abs/2108.11127)
## Auto-labeling Car Shape for KITTI
We release our Auto-labeling car shape data for KITTI with COCO formate. Each car instance has been assigned a 3D model. [Trainset](https://drive.google.com/file/d/1U6d4Z0l4FsAKUiv6jehT7esgsJ5ULWaI/view?usp=sharing) and [Valset](https://drive.google.com/file/d/1KfHiPOjWyV-pW3jxyTogzG07KjLvsF2g/view?usp=sharing) with  3000 vertexes 3D models annotations can be download from Google Drive. 


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
