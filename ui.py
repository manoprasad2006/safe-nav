# ui.py
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QTextEdit, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

import cv2

class SafeNavUI(QWidget):
    def __init__(self, start_callback):
        super().__init__()
        self.setWindowTitle("🛡️ SafeNav AI")
        self.setStyleSheet("background-color: #1e1e1e; color: white; font-family: Arial;")
        self.setFixedSize(900, 600)

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet("border: 2px solid #888;")

        self.log_area = QTextEdit(self)
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("background-color: #2b2b2b; color: #00ffcc;")

        self.start_button = QPushButton("▶ Start Detection")
        self.start_button.setStyleSheet("background-color: #00b894; color: white; font-weight: bold; padding: 10px;")
        self.start_button.clicked.connect(start_callback)

        self.fps_label = QLabel("FPS: 0", self)
        self.fps_label.setAlignment(Qt.AlignCenter)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.image_label)
        left_layout.addWidget(self.fps_label)
        left_layout.addWidget(self.start_button)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Logs:"))
        right_layout.addWidget(self.log_area)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)

    def update_image(self, frame, fps):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(image))
        self.fps_label.setText(f"FPS: {int(fps)}")

    def log(self, msg):
        self.log_area.append(msg)
