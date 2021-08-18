# AutoShape
ICCV2021 Paper: AutoShape: Real-Time Shape-Aware Monocular 3D Object Detection

## Auto-labeling Car Shape for KITTI
We release our Auto-labeling car shape data for KITTI with COCO formate. Each car instance has been assigned a 3D model. Annotations with full 668 vertexes 3D model can be download from here. 
### Data Formate
```python
# we add 2D/3D keypoints in KITTI car instance annotations
additional_instance_label = {
    'keypoints': list # 668 * 3 (u, v, visiblity),
    'p3d': list # 668 * 3 (x, y, z in model local coordinate)
}
```
