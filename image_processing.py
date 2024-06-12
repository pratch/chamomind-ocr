import cv2
import numpy as np

from utils import cv2_imshow_at_height

def fix_orientation(img):
    # TODO
    return img

def crop_text_region(img):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # convert to binary
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]

    # apply erosion to remove small noise/artifacts
    kernel = np.ones((3, 3), np.uint8)
    eroded_img = cv2.erode(thresh, kernel, iterations=1)

    # invert color
    eroded_img = cv2.bitwise_not(eroded_img)

    # apply dilation to merge texts into larger chunks
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated_img = cv2.dilate(eroded_img, rect_kernel, iterations=1)
    # cv2.imshow('dilation', dilated_img)

    # find contours
    contours, _ = cv2.findContours(
        dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find contour boundary (ink region)
    min_x, min_y, max_x, max_y = float('inf'), float(
        'inf'), float('-inf'), float('-inf')
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    '''  
    cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    cv2_imshow_at_height('cropped region', img, height=900)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    # crop the image
    cropped_img = img[min_y:max_y, min_x:max_x, :]
    return cropped_img
