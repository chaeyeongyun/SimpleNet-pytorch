# SimpleNet-pytorch
Li, Jichun, Weimin Tan, and Bo Yan. "Perceptual variousness motion deblurring with light global context refinement." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021. Pytorch Implementation  
https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Perceptual_Variousness_Motion_Deblurring_With_Light_Global_Context_Refinement_ICCV_2021_paper.pdf

---

### model architecture

![image](https://user-images.githubusercontent.com/70565663/173803742-8c2a2398-d154-48f5-aeb7-b43c8f2ecedd.png)  
![image](https://user-images.githubusercontent.com/70565663/173981900-65d983d0-0943-4b37-8a0a-8b8b81187429.png)
![image](https://user-images.githubusercontent.com/70565663/173981978-d93e8e6f-9300-43b0-8f69-68f4128e287a.png)

### Train

```python
python train.py --config ./config/config_file.yaml 
```

### Test

```python
python test.py --data_dir ./testimages --save_dir ./test --weights ./train/ckpoints/model.pth
```
