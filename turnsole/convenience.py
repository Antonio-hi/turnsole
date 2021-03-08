import cv2
import numpy as np

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

def resize_with_pad(image, side_length):
    # 保持宽高比，将图像长边缩短或填充至指定长度，短边等比例缩放后再填充至指定长度
    height, width, _ = image.shape
    max_side = max(height, width)
    ratio = side_length / max_side if max_side > side_length else 1.

    resized = cv2.resize(image, (0, 0), fx=ratio, fy=ratio)
    h, w, _ = resized.shape
    canvas = np.ones((side_length, side_length, 3), image.dtype)
    canvas[:h, :w, :] = resized
    return canvas
