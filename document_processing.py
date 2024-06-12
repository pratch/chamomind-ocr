import pandas as pd
import utils
from utils import DocumentTypes


def identify_doc_type(img, ocr_results):
    has_juris_record_title = any(
        item[1] == 'หนังสือรับรอง' for item in ocr_results)
    has_juris = any('นิติบุคคล' in item[1] for item in ocr_results)
    print(has_juris_record_title, has_juris)
    is_juris_record = has_juris_record_title and has_juris

    if is_juris_record:
        return DocumentTypes.JURIS_RECORD
    else:
        return DocumentTypes.UNKNOWN


def extract_fields(img, ocr_results, doc_type):

    # TODO: return dict of important fields
    return []
