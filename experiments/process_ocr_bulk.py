from pathlib import Path
import cv2
import json
import pickle
from utils import cv2_imshow_at_height
from PIL import Image, ImageFont, ImageDraw
import numpy as np

dataset_path = 'orig_doc_by_type'
output_folder = 'out'

print(len(list(Path(output_folder).rglob('*.json'))))

viewed_types = []
for json_path in Path(output_folder).rglob('*.json'):
    if json_path.parent.name in viewed_types: # show only one per doc type
        continue
    print(json_path.parent.name)
    viewed_types.append(json_path.parent.name)

    filename = json_path.stem
    image_path = json_path.parent / f'{filename}_clean.png'
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = ~img

    f = open(json_path, "r", encoding='utf-8')  
    results = json.load(f)

    # draw boxes
    for result in results:
        text = result['text']
        confidence = result['confidence']
        leftcorner = result['bbox_left']
        rightcorner = result['bbox_right']
        cv2.rectangle(img, leftcorner, rightcorner, (0, 255, 0), 2)

    # draw detected texts
    pil_image = Image.fromarray(img)
    fontpath = './angsana.ttf'
    font = ImageFont.truetype(fontpath,30)

    for idx, result in enumerate(results):
        text = result['text']
        confidence = result['confidence']
        leftcorner = result['bbox_left']
        draw = ImageDraw.Draw(pil_image)
        draw.text((leftcorner[0], leftcorner[1] - 20),text,font=font,fill=(255,0,0))

    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2_imshow_at_height('ocr result',img, 900)
    cv2.waitKey(0)
