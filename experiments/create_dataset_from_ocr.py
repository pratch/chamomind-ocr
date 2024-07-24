from pathlib import Path
import json
from utils import cv2_imshow_at_height
import numpy as np
import pandas as pd

dataset_path = 'orig_doc_by_type'
output_folder = 'out'

print(len(list(Path(output_folder).rglob('*.json'))))

doc_type_past = ''
dataset = []
for json_path in Path(output_folder).rglob('*.json'):
    filename = json_path.stem
    doc_type = json_path.parent.name

    if doc_type == 'orieatation':
        continue

    if doc_type != doc_type_past:
        print('===========================')
        print('doc_type: ' + doc_type)

    doc_type_past = doc_type

    f = open(json_path, "r", encoding='utf-8')  
    results = json.load(f)

    appended_text = ""
    for result in results:
        text = result['text']
        confidence = result['confidence']
        leftcorner = result['bbox_left']
        rightcorner = result['bbox_right']
        if confidence > 0.4 and len(text) >= 5:
            appended_text += text + ' '
    
    if len(appended_text) > 0:
        dataset.append((appended_text, doc_type)) # create dataset of text + label

    #    print('>>>', appended_text)
    #else:
    #    print('skipped')
    
columns = ['text', 'doc_type']
df = pd.DataFrame(dataset, columns=columns)
df.to_csv('document_classification_dataset.csv', index=False, encoding='utf-8')    
