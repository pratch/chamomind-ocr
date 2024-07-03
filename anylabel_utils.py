
import json
import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
from image_processing import crop_text_region
from utils import cv2_imshow_at_height


def parse_anylabeling_json(dataset_path):
    """
    Parses the JSON files in the specified dataset path and extracts label data including filename, document type, image path, height, width, and bounding box coordinates.

    Parameters:
    - dataset_path (str): The path to the dataset containing JSON files.

    Returns:
    - parsed_jsons (list): A list of dictionaries containing the extracted label data for each JSON file in the dataset.
    """
    parsed_jsons = []
    for json_path in Path(dataset_path).rglob('*.json'): # get all json file paths
        # convert json to dict
        label_data = {}
        label_data['filename'] = json_path.stem
        label_data['doc_type'] = json_path.parent.name
        label_data['json_path'] = json_path
        label_data['image_path'] = json_path.parent / f'{json_path.stem}.png'
        with open(json_path, encoding='utf-8') as f:
            label_json = json.load(f)
            label_data['height'] = label_json['imageHeight']
            label_data['width']  = label_json['imageWidth']
            bbox_data = []
            for l in label_json['shapes']:
                #print(l['label'], l['text'])
                points = np.array(l['points']) # can be rect or polygon
                x_arr = [np.round(p[0]).astype(int) for p in points]
                y_arr = [np.round(p[1]).astype(int) for p in points]
                x1 = min(x_arr) # convert polygon to enclosing rect
                y1 = min(y_arr)
                x2 = max(x_arr)
                y2 = max(y_arr)
                #normed_p1, normed_p2 = np.round(p1/[w, h], 3), np.round(p2/[w, h], 3)
                bbox_data.append({
                    'field': l['label'],
                    'text': l['text'],
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                })
            label_data['bboxes'] = bbox_data
            parsed_jsons.append(label_data)

    return parsed_jsons


def crop_images_and_adjust_bboxes(parsed_jsons):
    """
    Crop images to fit content and adjust bounding boxes for each JSON data in the parsed_jsons list.
    
    Args:
        parsed_jsons (list): A list of dictionaries containing JSON data for each image.
    """
    for json_data in parsed_jsons:
        # crop image and save as
        image_path = json_data['image_path']
        out_path = image_path.parent / f'{image_path.stem}_cropped.png'
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cropped_img, (min_x, min_y), (max_x, max_y) = crop_text_region(img)
        cv2.imwrite(str(out_path), cropped_img)
        
        # adjust bboxes
        json_data['height'] = max_y - min_y
        json_data['width'] = max_x - min_x
        for bbox in json_data['bboxes']:
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
            x2, y2 = int(bbox['x2']), int(bbox['y2'])
            bbox['x1'] = x1 - min_x
            bbox['y1'] = y1 - min_y
            bbox['x2'] = x2 - min_x
            bbox['y2'] = y2 - min_y
            

def compute_bbox_stats(parsed_jsons):
    """
    Compute statistics on bounding box data from parsed JSONs.
    
    Parameters:
    - parsed_jsons (list): A list of dictionaries containing parsed JSON data.
    """
    bbox_data = []
    for json_data in parsed_jsons:
        h = json_data['height']
        w = json_data['width']    
        for bbox in json_data['bboxes']:
            field_name = bbox['field']
            x1, y1 = bbox['x1'], bbox['y1']
            x2, y2 = bbox['x2'], bbox['y2']
            normed_x1 = round(x1/w, 3)
            normed_y1 = round(y1/h, 3)
            normed_x2 = round(x2/w, 3)
            normed_y2 = round(y2/h, 3)
            bbox_data.append({
                'filename': json_data['filename'],
                'doc_type': json_data['doc_type'],
                'field': field_name,
                'normed_x1': normed_x1,
                'normed_y1': normed_y1,
                'normed_x2': normed_x2,
                'normed_y2': normed_y2
            })
    df = pd.DataFrame(bbox_data)
    print('== average bbox positions ==')
    print(df.groupby(['doc_type','field']).mean(numeric_only=True))
    print('== stdev bbox positions ==')
    print(df.groupby(['doc_type','field']).std(numeric_only=True, ddof=0)) # population stdev to avoid NaN     
    
    # TODO: return ?
                    

if __name__ == "__main__":
    dataset_path = 'orig_doc_by_type'

    parsed_jsons = parse_anylabeling_json(dataset_path)
    #print(len(parsed_jsons))
    
    crop_images_and_adjust_bboxes(parsed_jsons)
    
    # print bbox stats by doc type and field (average position, stdev)
    compute_bbox_stats(parsed_jsons)
    
    # TODO: convert anylabel to fields_position.csv (using average positions)
    