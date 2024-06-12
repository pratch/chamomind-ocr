import cv2


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

def cv2_imshow_at_height(winname, img, height=900):
    h, w, _ = img.shape
    new_w = int(w*height/h)
    resized_img = cv2.resize(img, (new_w, height))
    cv2.imshow(winname, resized_img, height)
