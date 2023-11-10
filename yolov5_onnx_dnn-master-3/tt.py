import sys
import os
import shutil

from PyQt5.QtGui import QImage, QPixmap
from yolov5_dnn import yolov5
from yolov5_dnn import mult_test
import subprocess
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QSlider
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl, QTimer
import cv2


class RealtimeDetection(QWidget):
    def __init__(self):
        super().__init__()


        # 创建媒体播放器和视频显示部件
        self.player = QMediaPlayer(self)
        self.video_widget = QVideoWidget(self)
        self.player.setVideoOutput(self.video_widget)

        self.label = QLabel(self)
        self.label.hide()
        # self.label.setAlignment(Qt.AlignCenter)
        # self.label.setAlignment(Qt.AlignVCenter)
        self.label.setGeometry(100,1,400,500)

        # self.label.setFixedSize(400,300)


        self.setFixedSize(640, 480)

        # 创建控制按钮和进度条
        self.open_button = QPushButton("打开视频", self)
        self.play_button = QPushButton("播放", self)
        self.pause_button = QPushButton("暂停", self)
        self.progress_bar = QSlider(Qt.Horizontal, self)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.sliderMoved.connect(self.set_position)
        self.detect_button = QPushButton("开始识别", self)
        self.realtime_detect_button = QPushButton("实时识别", self)
        self.stop_realtime_detection=QPushButton("停止实时识别",self)

        # 设置布局
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.open_button)
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.detect_button)
        button_layout.addWidget(self.realtime_detect_button)
        button_layout.addWidget(self.stop_realtime_detection)
        # button_layout.addWidget(self.label, stretch=1)  # 将self.label添加到布局中，并设置stretch参数

        layout = QVBoxLayout()
        layout.addLayout(button_layout)
        layout.addWidget(self.video_widget)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

        # 信号与槽连接
        self.open_button.clicked.connect(self.open_video)
        self.play_button.clicked.connect(self.play_video)
        self.pause_button.clicked.connect(self.player.pause)
        self.player.durationChanged.connect(self.progress_bar.setMaximum)
        self.player.positionChanged.connect(self.progress_bar.setValue)
        self.realtime_detect_button.clicked.connect(self.start_realtime_detection)
        self.detect_button.clicked.connect(self.start_detection)
        self.stop_realtime_detection.clicked.connect(self.stop_recognition)

        # 存储视频文件的路径
        self.video_path = ""

        # 打开摄像头
        # self.cap = cv2.VideoCapture(0)  # 0表示默认摄像头



    def open_video(self):
        # 打开视频文件
        file_path = QFileDialog.getOpenFileName(self, "选择视频文件")[0]

        if file_path:
            media = QMediaContent(QUrl.fromLocalFile(file_path))
            self.player.setMedia(media)
            self.player.play()
            # 将视频文件路径存储在self.video_path中
            self.video_path = file_path

    def set_position(self, position):
        self.player.setPosition(position)

    def play_video(self):
        # 如果当前是停止状态，设置新的视频路径并播放
        if self.player.state() == QMediaPlayer.StoppedState:
            video_path = "/Volumes/Hard_disk/yolov5_onnx_dnn-master-3/output_image/text.mp4"
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
        self.player.play()

    def start_detection(self):
        # 创建input_image文件夹用于存放输入视频
        input_folder = "input_image"
        os.makedirs(input_folder, exist_ok=True)

        # 将选定的视频复制到input_image文件夹中，保存为text.mp4
        input_video_path = os.path.join(input_folder, 'text.mp4')
        shutil.copyfile(self.video_path, input_video_path)

        onnx_path = r'./yolov5s.onnx'
        input_path = r'./input_image'
        save_path = r'./output_image'

        mult_test(onnx_path, input_path, save_path, video=False)
        video_path = "/Volumes/Hard_disk/yolov5_onnx_dnn-master-3/output_image/text.mp4"
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
        self.player.play()

    def realtime_detection(self):
        self.cap = cv2.VideoCapture(0)  # 0表示默认摄像头
        onnx_path = r'./yolov5s.onnx'
        model = yolov5(onnx_path)

        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = self.cap.get(cv2.CAP_PROP_FPS)  # 视频平均帧率
        size = (frame_height, frame_width)  # 尺寸和帧率和原视频相同
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('zi.mp4', fourcc, fps, size)

        ret, frame = self.cap.read()
        if not ret:
            print("无法读取")
        else:
            frame = model.detect(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = QImage(frame.data, frame_width, frame_height, QImage.Format_RGB888)
            scaled_image = image.scaled(self.label.size(), Qt.KeepAspectRatio)

            self.label.setPixmap(QPixmap.fromImage(scaled_image))
            self.label.show()  # 显示self.label

    def start_realtime_detection(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.realtime_detection)
        self.timer.start(30)

    def stop_recognition(self):
        self.label.clear()
        self.label.hide()
        self.timer.stop()
        self.cap.release()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = RealtimeDetection()
    player.show()
    sys.exit(app.exec_())
