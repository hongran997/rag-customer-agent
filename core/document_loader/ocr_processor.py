import io
from utils.logger import logger

_ocr_reader = None
_ocr_failed = False

OCR_IMAGE_MIN_SIZE = 64
OCR_CONFIDENCE_THRESHOLD = 0.3


def _get_reader():
    global _ocr_reader, _ocr_failed
    if _ocr_failed:
        return None
    if _ocr_reader is None:
        logger.info("首次使用 EasyOCR，正在加载模型（仅加载一次）...")
        try:
            import easyocr
            _ocr_reader = easyocr.Reader(
                ["ch_sim", "en"],
                gpu=False,
                verbose=False,
            )
            logger.info("EasyOCR 模型加载完成")
        except Exception as e:
            _ocr_failed = True
            logger.warning(f"EasyOCR 初始化失败，模型不可用: {e}")
            return None
    return _ocr_reader


def _load_image_as_array(image_bytes: bytes):
    from PIL import Image
    import numpy as np
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def ocr_image_bytes(image_bytes: bytes) -> str:
    reader = _get_reader()
    if reader is None:
        return ""
    try:
        img_array = _load_image_as_array(image_bytes)
        results = reader.readtext(img_array)
        texts = []
        for bbox, text, confidence in results:
            if confidence >= OCR_CONFIDENCE_THRESHOLD:
                text = text.strip()
                if text:
                    texts.append(text)
        return " ".join(texts)
    except Exception as e:
        logger.warning(f"OCR 识别图片失败: {e}")
        return ""


def should_skip_image(width: int, height: int) -> bool:
    return width < OCR_IMAGE_MIN_SIZE or height < OCR_IMAGE_MIN_SIZE
