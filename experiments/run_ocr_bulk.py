import easyocr
from pathlib import Path
import cv2
import json
import pickle

dataset_path = 'orig_doc_by_type'
output_folder = 'out'

languages = ['th', 'en']
reader = easyocr.Reader(languages)

processed = []

for image_path in Path(dataset_path).rglob('*.png'):
  img = cv2.imread(str(image_path))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.GaussianBlur(img, (5, 5), 0)
  _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  ocr_results = reader.readtext(img)

  json_data = []
  for detection in ocr_results:
    bounding_box, text, confidence = detection
    leftcorner = tuple(map(int, bounding_box[0]))
    rightcorner = tuple(map(int, bounding_box[2]))
    json_data.append({
      "text": text,
      "confidence": confidence,
      "bbox_left": leftcorner,
      "bbox_right": rightcorner
    })
  #cv2_imshow(img)
  json_path = output_folder / image_path.parent / f'{image_path.stem}.json'
  out_im_path = output_folder / image_path.parent / f'{image_path.stem}_clean.png'
  Path(output_folder / image_path.parent).mkdir(parents=True, exist_ok=True) # create folders
  with open(json_path, "w+", encoding='utf-8') as outfile: 
    json.dump(json_data, outfile)
  cv2.imwrite(str(out_im_path), img)

  processed.append(str(image_path)) # save processed file list
  with open('ocr_done.txt', 'wb') as f:
    pickle.dump(processed, f)
    
  print('saved result to ', json_path)