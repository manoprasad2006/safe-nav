# main.py
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal
from ui import SafeNavUI
from detector import ObjectDetector

import cv2

class DetectionThread(QThread):
    update_signal = pyqtSignal(object, float)
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.detector = ObjectDetector()

    def run(self):
        self.detector.detect(
            update_frame=lambda frame, fps: self.update_signal.emit(frame, fps),
            log_callback=lambda msg: self.log_signal.emit(msg)
        )

def main():
    app = QApplication(sys.argv)
    
    thread = DetectionThread()
    ui = SafeNavUI(start_callback=thread.start)  # ✅ Proper callback passed

    thread.update_signal.connect(ui.update_image)
    thread.log_signal.connect(ui.log)

    ui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
