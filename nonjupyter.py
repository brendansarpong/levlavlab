import sys
import cv2
import numpy as np
import os
import csv
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QProgressBar, QLineEdit, QMessageBox
)
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal


class VideoProcessor(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(np.ndarray, float, float)
    error = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit("Failed to open video.")
                return

            flow_speeds = []
            contours = []
            ret, prev = cap.read()
            prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for i in range(frame_count - 1):
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                flow_speeds.append(np.mean(mag))
                prev_gray = gray
                self.progress.emit(int((i / frame_count) * 100))

            avg_speed = np.mean(flow_speeds)
            mask = self.create_lava_mask(prev)
            width = self.calculate_flow_width(mask)
            self.finished.emit(mask, width, avg_speed)
        except Exception as e:
            self.error.emit(str(e))

    def create_lava_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 50, 50])
        upper = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower, upper)
        lower = np.array([160, 50, 50])
        upper = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_or(mask1, mask2)
        return mask

    def calculate_flow_width(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            return float(w)
        return 0.0


class LavaApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lava Flow Analyzer")
        self.setGeometry(100, 100, 500, 400)
        self.setStyleSheet("background-color: #F2F1F0;")

        font = QFont("Arial", 10)
        self.setFont(font)

        layout = QVBoxLayout()

        self.label = QLabel("Upload a video of lava flow:")
        layout.addWidget(self.label)

        self.upload_btn = QPushButton("Choose Video")
        self.upload_btn.clicked.connect(self.open_file)
        layout.addWidget(self.upload_btn)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        self.abg_input = QLineEdit()
        self.abg_input.setPlaceholderText("Enter Above Ground Level (AGL) in meters")
        layout.addWidget(self.abg_input)

        self.fov_input = QLineEdit()
        self.fov_input.setPlaceholderText("Enter Field of View (FOV) in degrees")
        layout.addWidget(self.fov_input)

        self.results = QLabel("")
        layout.addWidget(self.results)

        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        self.export_btn = QPushButton("Export Results as CSV")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_results)
        layout.addWidget(self.export_btn)

        self.setLayout(layout)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video File")
        if file_path:
            self.process_video(file_path)

    def process_video(self, video_path):
        self.processor = VideoProcessor(video_path)
        self.processor.progress.connect(self.progress.setValue)
        self.processor.finished.connect(self.show_results)
        self.processor.error.connect(self.show_error)
        self.processor.start()

    def show_results(self, mask, width, speed):
        height, width_mask = mask.shape
        q_img = QImage(mask.data, width_mask, height, width_mask, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img).scaled(400, 300, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

        self.results.setText(f"Flow Width: {width:.2f} m\nAverage Speed: {speed:.2f} m/s")
        self.export_data = {
            "width": width,
            "speed": speed,
            "agl": self.abg_input.text(),
            "fov": self.fov_input.text(),
            "timestamp": datetime.now().isoformat()
        }
        self.export_btn.setEnabled(True)

    def export_results(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "lava_results.csv", "CSV files (*.csv)")
        if file_path:
            with open(file_path, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.export_data.keys())
                writer.writeheader()
                writer.writerow(self.export_data)
            QMessageBox.information(self, "Success", "Results exported successfully.")

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LavaApp()
    window.show()
    sys.exit(app.exec_())
