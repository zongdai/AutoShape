# AutoShape
ICCV2021 Paper: AutoShape: Real-Time Shape-Aware Monocular 3D Object Detection

## Auto-labeling Car Shape for KITTI
We release our Auto-labeling car shape data for KITTI with COCO formate. Each car instance has been assigned a 3D model. [Trainset](https://drive.google.com/file/d/1mg4-Ved19Hy_e8w7opQ8srHNO8Dn4-W1/view?usp=sharing) and [Valset](https://drive.google.com/file/d/1emzXfEnXk8mzYPLAUlbw0z0vsp73SlbJ/view?usp=sharing) with full annotations 668 vertexes 3D model can be download from Google Drive. 
### Data Formate
```python
# we add 2D/3D keypoints in KITTI car instance annotations
additional_instance_label: 
    'keypoints_2D': list # 668 * 3 (u, v, visiblity),
    'keypoints_3D': list # 668 * 3 (x, y, z in model local coordinate)
```
