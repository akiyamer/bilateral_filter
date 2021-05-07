# bilateral_filter
Achieve a bilateral_filter function with python for the DIP course homework.

## 介绍

此项目包含了一个基础的双边滤波算法，使用python实现。

## 使用方式

### 基本用法
```python
import cv2

img = cv2.imread(r"original_img.jpg")
bf = BilateralFilter()
filtered_img = bf.filter(img)
```

### 设置滤波器半径、标准差
```python
"""
:param radius: 滤波器半径，滤波器大小为(2*radius+1)
:param sigma_r: 像素差异的高斯标准差
:param sigma_d: 空间距离的高斯标准差
"""
bf = BilateralFilter(3, 30, 50)
filtered_img = bf.filter(img)
```

