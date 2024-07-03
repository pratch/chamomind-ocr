import numpy as np
import pandas as pd
import utils
from utils import DocumentTypes, is_similar


def text_equivalent(text1, text2):
    # TODO: find levenshtein(t1,t2) <= 2
    return text1 == text2

def identify_doc_type(img, ocr_results):
    # TODO: support more doc types
    has_juris_record_title = any(text_equivalent(item[1], 'หนังสือรับรอง') for item in ocr_results)
    has_juris = any('นิติบุคคล' in item[1] for item in ocr_results)
    # print(has_juris_record_title, has_juris)
    is_juris_record = has_juris_record_title and has_juris
    

    has_juris_regis_title = any(is_similar(item[1], 'ใบสำคัญแสดงการจดทะเบียนห้างหุ้นส่วนบริษัท', 5) == 'ใบสำคัญแสดงการจดทะเบียนห้างหุ้นส่วนบริษัท' for item in ocr_results)
    has_juris_regis = any('กรมพัฒนาธรุกิจการค้า' in is_similar(item[1], 'กรมพัฒนาธรุกิจการค้า', 3) for item in ocr_results)
    is_has_juris_regis = has_juris_regis_title and has_juris_regis

    has_foreign_data_title = any(is_similar(item[1], 'ค้นหาข้อมูลคนต่างด้าว', 3) == 'ค้นหาข้อมูลคนต่างด้าว' for item in ocr_results)
    has_foreign_data = any('ข้อมูลคนต่างด้าว' in is_similar(item[1], 'ข้อมูลคนต่างด้าว', 3) for item in ocr_results)
    is_has_foreign_data = has_foreign_data_title and has_foreign_data
    

    if is_juris_record:
        return DocumentTypes.JURIS_RECORD
    elif is_has_juris_regis:
        return DocumentTypes.JURIS_REGIS
    elif is_has_foreign_data:
        return DocumentTypes.FOREIGN_DATA
    else:
        return DocumentTypes.UNKNOWN
    


def extract_fields(img, ocr_results, doc_type):
    # create list of ocr result corners
    ocr_result_corners = []
    for result in ocr_results:
        bounding_box = result[0]
        leftcorner = tuple(map(int, bounding_box[0]))
        height, width, _ = img.shape
        normed_leftcorner = (leftcorner[0]/width, leftcorner[1]/height)
        ocr_result_corners.append(normed_leftcorner)
    
    df = pd.read_csv('field_positions/' + doc_type + '_fields.csv')
    important_fields = []
    for idx, row in df.iterrows():
        target_field_corner = np.asarray((row.x1, row.y1))
        
        # find closest ocr corner
        # TODO: try finding ocr box with highest IoU with csv box instead?
        dist_2 = np.sum((ocr_result_corners - target_field_corner)**2, axis=1)
        closest_idx = np.argmin(dist_2)
        
        closest_ocr_result = ocr_results[closest_idx]
        # text = closest_ocr_result[1] # return just text instead of ocr result?
        
        important_fields.append((row.field, closest_ocr_result))

    return important_fields
