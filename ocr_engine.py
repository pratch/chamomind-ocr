import easyocr
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw


def ocr_easyocr(img):
    languages = ['th', 'en']
    reader = easyocr.Reader(languages)
    results = reader.readtext(img)
    return results


def draw_ocr_results(img, ocr_results):
    result_img = img.copy()

    # draw boxes
    for result in ocr_results:
        bounding_box = result[0]
        text = result[1]
        confidence = result[2]
        leftcorner = tuple(map(int, bounding_box[0]))
        rightcorner = tuple(map(int, bounding_box[2]))
        cv2.rectangle(result_img, leftcorner, rightcorner, (0, 255, 0), 2)

    # draw detected texts with PIL
    result_img = Image.fromarray(result_img)
    fontpath = './font/angsana.ttf'
    font = ImageFont.truetype(fontpath, 30)

    for idx, result in enumerate(ocr_results):
        bounding_box = result[0]
        text = result[1]
        confidence = result[2]
        leftcorner = tuple(map(int, bounding_box[0]))
        draw = ImageDraw.Draw(result_img)
        draw.text((leftcorner[0], leftcorner[1] - 20),
                  text, font=font, fill=(255, 0, 0))

    # convert back to opencv img
    result_img = cv2.cvtColor(np.array(result_img), cv2.COLOR_RGB2BGR)
    return result_img
