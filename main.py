import argparse
import cv2
from PIL import Image
from image_processing import crop_text_region


def main(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print('Cropping text region')
    cropped_img = crop_text_region(img)
    cv2.imwrite('cropped.png', cropped_img)

	# TODO: ocr, extract text
    print('Performing OCR')
    print(f'Extracted text from {image_path}:')


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
