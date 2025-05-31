# Loglama işlemleri için yardımcı fonksiyonlar 
import logging
import os

def get_logger(name: str):
    """Her modül için zaman damgalı, dosya ve konsol loglama desteği sağlar."""
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        # Konsol logu
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # Dosya logu
        log_dir = os.getenv("LOG_DIR", "logs")
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger 