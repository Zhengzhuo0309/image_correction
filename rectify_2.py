import cv2
import numpy as np


def rectify_video(image, map1, map2):
    frame_rectified = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return frame_rectified


def calculate_para(K, D, width, height):
    # 优化内参数和畸变系数
    p = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (width, height), None)
    # 此处计算花费时间较大，需从循环中抽取出来
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, None, p, (width, height), cv2.CV_32F)
    return map1, map2


def main():
    # K D 参数
    K = np.array(
        [[1201.3967181633418, 0.0, 909.7424436183744], [0.0, 1203.635467250557, 534.1590658991514], [0.0, 0.0, 1.0]])
    D = np.array([[-0.0978537375125563], [-0.03841501213366177], [-0.03612764818273854], [0.05276041355808103]])
    width, height = 1920, 1080
    # 参数优化
    map1, map2 = calculate_para(K, D, width, height)

    video = cv2.VideoCapture("20230314_2mm.mp4")
    out = cv2.VideoWriter('20230314_2mm_rectify.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 24, (width, height), True)
    while (video.isOpened()):
        success, image = video.read()
        if not success:
            break
        # 畸变矫正
        frame_rectified = rectify_video(image, map1, map2)
        # 将矫正后的帧写入到输出视频中
        out.write(frame_rectified)
        cv2.imshow('frame', frame_rectified)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()