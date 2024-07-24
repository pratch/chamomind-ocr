from anylabel_utils import parse_anylabeling_json, crop_images_and_adjust_bboxes, compute_bbox_stats

def match_bboxes(ocr_results, anylabel_jsons):

    return anylabel_jsons


if __name__ == "__main__":
    dataset_path = 'orig_doc_by_type'

    parsed_jsons = parse_anylabeling_json(dataset_path)
    crop_images_and_adjust_bboxes(parsed_jsons)

