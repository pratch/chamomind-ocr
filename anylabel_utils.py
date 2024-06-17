
import json
import cv2
import numpy as np
import os


def read_json(dataset_path):
    for doctype_folder in os.listdir(dataset_path):
        path = os.path.join(dataset_path, doctype_folder)
        for filename in os.listdir(path):
            if filename.lower().endswith('.json'):
                print(filename)
                with open(os.path.join(path, filename), encoding='utf-8') as f:
                    label_json = json.load(f)
                    h = label_json['imageHeight']
                    w = label_json['imageWidth']
                    for l in label_json['shapes']:
                        print(l['label'])
                        print(l['text'])
                        p1, p2 = np.array(l['points'])
                        normed_p1, normed_p2 = np.round(
                            p1/[w, h], 3), np.round(p2/[w, h], 3)
                        x1, y1 = np.round(p1).astype(int)
                        x2, y2 = np.round(p2).astype(int)
                        normed_x1, normed_y1 = normed_p1
                        normed_x2, normed_y2 = normed_p2
                        print(x1, y1)
                        print(normed_x1, normed_y1)
                        print(x2, y2)
                        print(normed_x2, normed_y2)


# convert anylabel to fields_position.csv
if __name__ == "__main__":
    dataset_path = 'orig_doc_by_type'

    jsons = read_json(dataset_path)
    # read anylabel

    # print bbox stats by doc type

    # find stats (average bbox positions)

    # for field_name, bbox in average_bboxes.items():
    #  print(f"{field_name}: {bbox}")
