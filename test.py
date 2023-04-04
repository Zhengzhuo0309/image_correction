import cv2
import numpy as np
import glob
import os

# 找棋盘格角点
# 棋盘格模板规格(内角点个数，内角点是和其他格子连着的点,如10 X 7)
w = 11
h = 8

flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE  # 11
flags_fisheye = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW  # 14

# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((1, w * h, 3), np.float32)
objp[0, :, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = []  # 在世界坐标系中的三维点
imgpoints = []  # 在图像平面的二维点

# 标定所用图像（路径不能有中文）
# images = glob.glob('./picture/pic1/*.jpg')
images = glob.glob('./317_1data/data/*.jpg')

size = tuple()
for fname in images:
    img = cv2.imread(fname)

    # 修改图像尺寸，参数依次为：输出图像，尺寸，沿x轴，y轴的缩放系数，INTER_AREA在缩小图像时效果较好
    # img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度
    size = gray.shape[::-1]  # 矩阵转置

    # 找到棋盘格角点
    # 棋盘图像(8位灰度或彩色图像)  棋盘尺寸  存放角点的位置
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    if ret:
        # 角点精确检测
        # criteria:角点精准化迭代过程的终止条件(阈值)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 400, 0.0000001)

        # 执行亚像素级角点检测
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners2)

        # 将角点在图像上显示
        cv2.drawChessboardCorners(img, (w, h), corners2, ret)
        cv2.imshow('findCorners', img)
        cv2.waitKey(10)

        print(fname)
    else:
        os.remove('%s' % fname)
    # print(fname)
# cv2.destroyAllWindows()

"""
标定、去畸变:
输入：世界坐标系里的位置 像素坐标 图像的像素尺寸大小 3*3矩阵，相机内参数矩阵 畸变矩阵
输出：标定结果 相机的内参数矩阵 畸变系数 旋转矩阵 平移向量
"""

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)

# mtx：内参数矩阵
# dist：畸变系数
# rvecs：旋转向量 （外参数）
# tvecs ：平移向量 （外参数）
print("ret:", ret)
print("内参数矩阵:\n", mtx, '\n')
print("畸变系数:\n", dist, '\n')

# 鱼眼/大广角镜头的单目标定
K = np.zeros((3, 3))
D = np.zeros((4, 1))
RR = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints))]
TT = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints))]
rms, _, _, _, _ = cv2.fisheye.calibrate(
    objpoints, imgpoints, gray.shape[:2][::-1], K, D, RR, TT, flags_fisheye, criteria
)
# 摄像头内参,此结果与mtx相比更为稳定和精确
print("K=np.array( " + str(K.tolist()) + " )")
# 畸变系数D = (k1,k2,k3,k4)
print("D=np.array( " + str(D.tolist()) + " )")

# 计算反投影误差
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
print("total error: ", mean_error / len(objpoints))

img2 = cv2.imread('1.jpg')
width, height = img2.shape[:2][::-1]
print(width, height)
# 优化内参数和畸变系数
p = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (width, height), None)
# 此处计算花费时间较大，需从循环中抽取出来
mapx2, mapy2 = cv2.fisheye.initUndistortRectifyMap(K, D, None, p, (width, height), cv2.CV_32F)

# 畸变矫正
frame_rectified = cv2.remap(img2, mapx2, mapy2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
cv2.imwrite('frame_rectified1.jpg', frame_rectified)


