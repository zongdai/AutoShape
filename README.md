# AutoShape
ICCV2021 Paper: AutoShape: Real-Time Shape-Aware Monocular 3D Object Detection

## Auto-labeling Car Shape for KITTI
We release our Auto-labeling car shape data for KITTI with COCO formate. Each car instance has been assigned a 3D model. [Trainset](https://drive.google.com/file/d/1tcb6m10kmC4v3-9mP8o68_XrVMD8EOUI/view?usp=sharing) and [Valset](https://drive.google.com/file/d/1X_CG3y6j0GxRXRxqCtUiZk-Zg1gn_nSi/view?usp=sharing) with annotations 3000 vertexes 3D model can be download from Google Drive. 
### Data Formate
```python
# we add 2D/3D keypoints in KITTI car instance annotations
additional_instance_label: 
    '2dkeypoints': list # (3000 + 9) * 3 (u, v, visiblity),
    '3dkeypoints': list # (3000 + 9) * 3 (x, y, z in model local coordinate)
```
