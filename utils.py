import Levenshtein
import cv2
import numpy as np


class DocumentTypes:
    UNKNOWN = 'unknown'
    COM_REGIS = 'com_regis'
    JURIS_RECORD = 'juris_record'
    JURIS_REGIS = 'juris_regis'
    FOREIGN_DATA = 'foreign_data'
    EMPLOYER_ACK = 'employer_ack'
    FOREIGN_ACK = 'foreign_ack'
    ENTRY_FORM = 'entry_form'
    EXIT_FORM = 'exit_form'
    PAY_RECEIPT = 'pay_receipt'
    PERMIT39 = 'permit39'
    EWORK = 'ework'
    PERMIT50 = 'permit50'
    APP_RECEIPT = 'app_receipt'
    PASSPORT = 'passport'
    VISA = 'visa'
    CITIZEN_ID = 'citizen_id'
    HOUSE_REG = 'house_reg'
    MED_CERT = 'med_cert'
    HEALTH_INSUR = 'health_insur'
    IMMIGRATE = 'immigrate'


DOC_TYPE_TH = {
    DocumentTypes.COM_REGIS: 'ทะเบียนพาณิชย์',
    DocumentTypes.JURIS_RECORD: 'ทะเบียนนิติบุคคล',
    DocumentTypes.JURIS_REGIS: 'ใบจดทะเบียนนิติ',
    DocumentTypes.FOREIGN_DATA: 'ใบข้อมูลคนต่างด้าว',
    DocumentTypes.EMPLOYER_ACK: 'ใบตอบรับนายจ้าง',
    DocumentTypes.FOREIGN_ACK: 'ใบตอบรับคนต่างด้าว',
    DocumentTypes.ENTRY_FORM: 'แบบแจ้งเข้า',
    DocumentTypes.EXIT_FORM: 'แบบแจ้งออก',
    DocumentTypes.PAY_RECEIPT: 'ใบเสร็จรับเงิน',
    DocumentTypes.PERMIT39: 'บต.39',
    DocumentTypes.EWORK: 'e-work',
    DocumentTypes.PERMIT50: 'บต.50',
    DocumentTypes.APP_RECEIPT: 'ใบรับขอและใบเสร็จ',
    DocumentTypes.PASSPORT: 'พาสปอร์ต',
    DocumentTypes.VISA: 'วีซ่า',
    DocumentTypes.CITIZEN_ID: 'บัตรประชาชน',
    DocumentTypes.HOUSE_REG: 'ทะเบียนบ้าน',
    DocumentTypes.MED_CERT: 'ใบรับรองแพทย์',
    DocumentTypes.HEALTH_INSUR: 'บัตรประกันสุขภาพ',
    DocumentTypes.IMMIGRATE: 'ตม.6',
}

search_words = {
    "ค้นหาข้อมูลคนต่างด้าว": 3, #foreign_data
    "ข้อมูลคนต่างด้าว": 3, #foreign_data
    "กรมพัฒนาธุรกิจการค้า": 3, # juris_regis
    "ใบสำคัญแสดงการจดทะเบียนห้างหุ้นส่วน": 5, # juris_regis
}

def cv2_imshow_at_height(winname, img, height=900):
    h, w, _ = img.shape
    new_w = int(w*height/h)
    resized_img = cv2.resize(img, (new_w, height))
    cv2.imshow(winname, resized_img)

def convert_polygon_to_rect(points):
    x_arr = [np.round(p[0]).astype(int) for p in points]
    y_arr = [np.round(p[1]).astype(int) for p in points]
    x1 = min(x_arr) # convert polygon to enclosing rect
    y1 = min(y_arr)
    x2 = max(x_arr)
    y2 = max(y_arr)
    return x1, y1, x2, y2

# ฟังก์ชันคำนวณ Levenshtein distance
def is_similar(target, text, threshold):
    distance = Levenshtein.distance(target, text)
    if distance <= threshold:
        return text
    else:
        return target
