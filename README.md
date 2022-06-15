# SimpleNet-pytorch
Li, Jichun, Weimin Tan, and Bo Yan. "Perceptual variousness motion deblurring with light global context refinement." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021. Pytorch Implementation  
https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Perceptual_Variousness_Motion_Deblurring_With_Light_Global_Context_Refinement_ICCV_2021_paper.pdf

---

### model architecture

![image](https://user-images.githubusercontent.com/70565663/173803742-8c2a2398-d154-48f5-aeb7-b43c8f2ecedd.png)  

### Train

```python
python train.py --config ./config/config_file.yaml 
```

### Test

```python
python test.py --data_dir ./testimages --save_dir ./test --weights ./train/ckpoints/model.pth
```
