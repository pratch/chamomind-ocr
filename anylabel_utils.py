
import json
import cv2
import numpy as np
import pandas as pd
import csv
import os
from pathlib import Path
from image_processing import crop_text_region
from utils import convert_polygon_to_rect, cv2_imshow_at_height


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
                points = l['points']
                x1, y1, x2, y2 = convert_polygon_to_rect(points)
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

    print('== stdev bbox positions but more 0.01 ==')
    std_df = df.groupby(['doc_type', 'field']).std(numeric_only=True, ddof=0)
    filtered_std_df = std_df[std_df >= 0.01].dropna()
    print(filtered_std_df)

    return df   


def convert_to_fields_position_csv(parsed_jsons, output_dir):

    df = compute_bbox_stats(parsed_jsons)
    
    avg_df = df.groupby(['doc_type', 'field']).mean(numeric_only=True).reset_index()

    # Round to 4 decimal places
    avg_df = avg_df.round({'normed_x1': 4, 'normed_y1': 4})

    # Rename normed_x1, normed_y1
    avg_df = avg_df.rename(columns={'normed_x1': 'x1', 'normed_y1': 'y1'})
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Group by doc_type and save each group to a separate CSV file
    for doc_type, group in avg_df.groupby('doc_type'):
        group = group[['field', 'x1', 'y1']]
        output_csv = os.path.join(output_dir, f'{doc_type}_fields.csv')
        group.to_csv(output_csv, index=False)
        print(f'Successfully saved to {output_csv}')

def convert_to_fields_fix_position_csv(parsed_jsons, output_dir):

    # Define fields for each doc_type
    fields_by_doc_type = {
        'app_receipt': ['doc_title', 'signature'],
        'employer_ack': ['doc_title', 'employer_field', 'employer_name'],
        'ework': ['doc_title_th', 'doc_title_en', 'employee_name_field_th', 'employee_name_field_en', 'employee_name', 
                  'foreign_id_field_th', 'foreign_id_field_en', 'foreign_id', 'passport_id_field_th', 'passport_id_field_en',
                  'passport_id', 'employ_cert_id_field_th', 'employ_cert_id_field_en', 'employ_cert_id', 'employ_end_date_field_th',
                  'employ_end_date_field_en', 'employ_end_date'],
        'foreign_ack': ['doc_title', 'foreign_field', 'foreign_name'],
        'foreign_data': ['doc_title', 'foreign_data', 'foreign_id_field', 'foreign_id', 'full_name_field', 
                         'full_name', 'address_field', 'address', 'nationality_field', 'nationality',
                         'job_type_field', 'job_type', 'passport_id_field', 'passport_id', 'sex_field', 'sex',
                         'cert_id_field', 'cert_id', 'cert_status_field', 'cert_status', 'expire_date_filed',
                         'expire_date', 'health_check_field', 'health_check', 'id_expire_date_field', 'id_expire_date',
                         'visa_expire_date_field', 'visa_expire_date', 'passport_expire_date_field', 'passport_expire_date', 'register_date_field',
                         'register_date', 'birth_date_field', 'birth_date', 'employer_field', 'employer',
                         'business_type_field', 'business_type', 'work_address_field', 'work_address'],
        'immigrate': ['doc_title_en', 'doc_title_th', 'departure_card_th', 'departure_card_en','admit_date_field', 
                      'admit_date', 'valid_until_date_field', 'valid_until_date', 'immigrate_id'],
        'juris_record': ['doc_title','company_field', 'company_name', 'committee_num', 'capital_field',
                         'capital_amount', 'head_office_field', 'head_office_address', 'page_no'],
        'juris_regis': ['regis_id_field', 'regis_id', 'form_title', 'dbd_title', 'doc_title', 
                        'cert_purpose_text', 'company_name', 'juris_declare_text', 'regis_date', 'issue_date'],
        'passport':['passport_id', 'first_name', 'last_name', 'nationality', 'birth_date', 
                    'sex', 'issue_date', 'expire_date', 'birth_place'],
        'pay_receipt':['doc_title', 'payer_field', 'payer_name', 'employer_field', 'employer_name'],
        'permit50':['doc_title', 'foreign_id', 'cert_id', 'thai_name', 'passport_id', 
                    'employer_name', 'expire_date']
         

        # เพิ่มเติม doc_type และ fields ที่ต้องการ
    }

    df = compute_bbox_stats(parsed_jsons)
    avg_df = df.groupby(['doc_type', 'field']).mean(numeric_only=True).reset_index()

    # Round to 4 decimal places
    avg_df = avg_df.round({'normed_x1': 4, 'normed_y1': 4})

    # Rename normed_x1, normed_y1
    avg_df = avg_df.rename(columns={'normed_x1': 'x1', 'normed_y1': 'y1'})
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Group by doc_type and save each group to a separate CSV file
    for doc_type, group in avg_df.groupby('doc_type'):
        if doc_type in fields_by_doc_type:
            fields = fields_by_doc_type[doc_type]

            # Select only the desired columns
            group = group[group['field'].isin(fields)][['field', 'x1', 'y1']]
            output_csv = os.path.join(output_dir, f'{doc_type}_fields.csv')
            group.to_csv(output_csv, index=False)
            print(f'Successfully saved to {output_csv}')
        else:
            print(f'Error {doc_type}')
                    

if __name__ == "__main__":
    dataset_path = 'orig_doc_by_type'

    # กำหนดรายชื่อ fields ที่ต้องการ
    fields = ['company_field', 'company_name']

    parsed_jsons = parse_anylabeling_json(dataset_path)
    #print(len(parsed_jsons))

    crop_images_and_adjust_bboxes(parsed_jsons)
    
    # print bbox stats by doc type and field (average position, stdev)
    compute_bbox_stats(parsed_jsons)
    
    # TODO: convert anylabel to fields_position.csv (using average positions)
    #convert_to_fields_position_csv(parsed_jsons, 'field_positions')
    convert_to_fields_fix_position_csv(parsed_jsons, 'field_positions')

    
