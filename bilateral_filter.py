# -*- coding: utf-8 -*-
# @Author  : xuhongzhe
# @Time    : 2021-05-06
# @Function: Achieve a bilateral_filter function with python for the DIP course homework.

import copy
import cv2
import numpy as np


class BilateralFilter():
    def __init__(self, radius=3, sigma_r=30, sigma_d=50):
        """
        构造函数
        :param radius: 滤波器半径，滤波器大小为(2*radius+1)
        :param sigma_r: 像素差异的高斯标准差
        :param sigma_d: 空间距离的高斯标准差
        """
        self.radius = radius
        self.sigma_r = sigma_r
        self.sigma_d = sigma_d
        # 计算空间权重矩阵
        [space_weight, space_pos] = self._compute_space()
        self.space_weight = space_weight
        self.space_pos = space_pos
        # 缓存像素值
        self.pixel_values = self._compute_pixel()

    def _compute_space(self):
        # 将二维数组一维化方便计算
        # 空间权重
        space_weight = []
        # 每个权重值对应的位置
        space_pos = []
        # 高斯函数常量系数
        const = -0.5 / (self.sigma_d * self.sigma_d)
        for i in range(-self.radius, self.radius + 1):
            for j in range(-self.radius, self.radius + 1):
                space_weight.append(np.exp((i * i + j * j) * const))
                space_pos.append((i, j))
        return [space_weight, space_pos]

    def _compute_pixel(self):
        # 像素平方和缓存数组
        pixel_values = []
        # 高斯函数常量系数
        const = -0.5 / (self.sigma_r * self.sigma_r)
        for i in range(256):
            pixel_values.append(np.exp(i * i * const))
        return pixel_values

    def filter(self, img):
        """
        双边滤波器函数
        :param img: 图片信息三维数组(height, width, rgb)
        :return: 降噪后的图像数组(height, width, rgb)
        """
        filteredImg = copy.copy(img)
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        for row in range(imgHeight):
            for col in range(imgWidth):
                for channel in range(0, 2):
                    # 像素计算结果
                    val = 0
                    # 双边滤波器矩阵权重和
                    weight = 0
                    # 加权求和计算
                    for i in range(len(self.space_pos)):
                        [y, x] = self.space_pos[i]
                        y += row
                        x += col
                        if x < 0 or y < 0 or x >= imgWidth or y >= imgHeight:
                            tmp = 0
                        else:
                            # 原图像(x, y)处的像素值
                            tmp = img[y][x][channel]
                        # 计算双边滤波器矩阵在(x, y)处的值
                        w = np.float32(self.space_weight[i]) * np.float32(
                            self.pixel_values[np.abs(tmp - img[row][col][channel])])
                        val += tmp * w
                        weight += w

                    filteredImg[row][col][channel] = np.uint8(val / weight)

        return filteredImg


def main():
    bf1 = BilateralFilter()
    bf2 = BilateralFilter(7)
    bf3 = BilateralFilter(3, 40, 80)
    bf4 = BilateralFilter(15, 40, 80)
    img = cv2.imread(r"original_img.jpg")
    filteredImg1 = bf1.filter(img)
    filteredImg2 = bf2.filter(img)
    filteredImg3 = bf3.filter(img)
    filteredImg4 = bf4.filter(img)
    # cv2.imshow("new", filteredImg1)
    # cv2.imshow("old", img)
    cv2.imwrite("filtered_img(3,30,50).jpg", filteredImg1)
    cv2.imwrite("filtered_img(7,30,50).jpg", filteredImg2)
    cv2.imwrite("filtered_img(3,40,80).jpg", filteredImg3)
    cv2.imwrite("filtered_img(15,40,80).jpg", filteredImg4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()