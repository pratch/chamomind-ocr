import numpy as np
import cv2
from anylabel_utils import parse_anylabeling_json
from ocr_engine import ocr_easyocr

def eval(parsed_jsons, ocr_engine=ocr_easyocr):
    # TODO: fix orient/crop before ocr?
        
    for json_data in parsed_jsons:
        filename = json_data['filename']
        doc_type = json_data['doc_type']
        image_path = json_data['image_path']
        h = json_data['height']
        w = json_data['width']    
        
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ocr_results = ocr_engine(img)
        
        ocr_result_corners = []
        for result in ocr_results:
            bounding_box = result[0]
            leftcorner = tuple(map(int, bounding_box[0]))
            #height, width, _ = img.shape
            #normed_leftcorner = (leftcorner[0]/width, leftcorner[1]/height)
            ocr_result_corners.append(leftcorner)
        
        for bbox in json_data['bboxes']:
            field_name = bbox['field']
            text_true = bbox['text']
            x1, y1 = bbox['x1'], bbox['y1']
            x2, y2 = bbox['x2'], bbox['y2']
            
            # TODO: ignore variable height fields for now (use compute_bbox_stats)?
            
            # create list of ocr result corners
            target_field_corner = np.asarray((x1, y1))
        
            # find closest ocr corner
            dist_2 = np.sum((ocr_result_corners - target_field_corner)**2, axis=1)
            closest_idx = np.argmin(dist_2)
        
            closest_ocr_result = ocr_results[closest_idx]
            text_pred = closest_ocr_result[1] 
            
            print('field:', field_name)
            print('\t text_true:', text_true)
            print('\t text_pred:', text_pred)
            
        break

if __name__ == '__main__':
    dataset_path = 'orig_doc_by_type'
    parsed_jsons = parse_anylabeling_json(dataset_path)

    eval(parsed_jsons, ocr_engine=ocr_easyocr)
