import argparse
import cv2
import utils
from PIL import Image
from image_processing import crop_text_region, fix_orientation
from document_processing import extract_fields, identify_doc_type
from ocr_engine import ocr_easyocr, draw_ocr_results

def main(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # TODO: fix document orientation
    print('Fixing document orientation')
    img = fix_orientation(img)

    print('Cropping text region')
    cropped_img, _, _ = crop_text_region(img)
    cv2.imwrite('cropped.png', cropped_img)
    
    # ocr, extract all texts 
    # TODO: add pytesseract, etc.
    print('Performing OCR')
    ocr_results = ocr_easyocr(cropped_img)
    ocr_img = draw_ocr_results(cropped_img, ocr_results)
    cv2.imwrite('raw_ocr_result.png', ocr_img)
    
    # TODO: post-process ocr results (mispelling correction) 
    
    # TODO: identify doc type based on ocr results
    doc_type = identify_doc_type(cropped_img, ocr_results)
    print('Detected document type:', utils.DOC_TYPE_TH[doc_type])
    
    # TODO: field extraction 
    # - read field_positions
    # - do some smart filtering for each file type
    # - line height detection for variable lines?
    extracted_fields = extract_fields(cropped_img, ocr_results, doc_type)
    
    # print final ocr results
    print(f'Extracted text from {image_path}:')
    for field in extracted_fields:
        field_name = field[0]
        field_ocr_result = field[1]
        field_text = field_ocr_result[1]
        print(f'{field_name}: {field_text}')
    
    filtered_ocr_results = [item[1] for item in extracted_fields]
    
    # create image with bbox of extracted fields only
    filtered_ocr_img = draw_ocr_results(cropped_img, filtered_ocr_results)
    cv2.imwrite('filtered_ocr_result.png', filtered_ocr_img)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chamomind OCR')
    parser.add_argument('image_file', type=str,
                        help='Path to the image file (must be .png).')
    # parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output.')
    args = parser.parse_args()

    if not args.image_file.lower().endswith('.png'):
        print('Error: Please provide a PNG image file.')
        exit()

    main(args.image_file)
