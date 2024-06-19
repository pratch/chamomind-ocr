import cv2
import numpy as np

from utils import cv2_imshow_at_height

def fix_orientation(img):
    # TODO
    return img

def crop_text_region(img):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # convert to binary (also remove noise with lighter color than threshold)
    # otsu threshold suitable for different light conditions
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # remove ink blobs
    cleaned = cv2.medianBlur(thresh, 11)
    cleaned = cv2.medianBlur(cleaned, 5) # TODO: better way to remove tiny remaining inks?

    # merge text into larger chunks, also give breathing room on crop borders
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated_img = cv2.dilate(cleaned, rect_kernel, iterations=1)
    # cv2.imshow('dilation', dilated_img)

    # find contours
    contours, _ = cv2.findContours(
        dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find contour boundary (text region)
    min_x, min_y, max_x, max_y = float('inf'), float(
        'inf'), float('-inf'), float('-inf')
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        # no need to crop right/bottom sides
        max_y, max_x = img.shape[:2] 
        #max_x = max(max_x, x + w) 
        #max_y = max(max_y, y + h)

    '''  
    cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    cv2_imshow_at_height('cropped region', img, height=900)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    # crop the image
    cropped_img = img[min_y:max_y, min_x:max_x, :]
    return (cropped_img, (min_x, min_y), (max_x, max_y))
