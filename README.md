# SimpleNet-pytorch
Li, Jichun, Weimin Tan, and Bo Yan. "Perceptual variousness motion deblurring with light global context refinement." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021. Pytorch Implementation  
https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Perceptual_Variousness_Motion_Deblurring_With_Light_Global_Context_Refinement_ICCV_2021_paper.pdf

---

### model architecture

![Untitled](%E1%84%89%E1%85%A6%E1%84%86%E1%85%B5%E1%84%82%E1%85%A1%20%E1%84%87%E1%85%A1%E1%86%AF%E1%84%91%E1%85%AD%203f2867c58cf2469e9feec0f30a446ade/Untitled.png)

### Train

```python
python train.py --config ./config/config_file.yaml 
```

### Test

```python
python test.py --data_dir ./testimages --save_dir ./test --weights ./train/ckpoints/model.pth
```
