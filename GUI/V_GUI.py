import os
import time
import sys
import cv2
import torch
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from cv2 import VideoCapture, CAP_PROP_FPS
from detect_plate import detect_Recognition_plate, draw_result, init_model, load_model, get_second

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detect_model = load_model('../weights/plate_rec.pt', device)
plate_rec_model = init_model(device, '../weights/plate_rec_color.pth', is_color=True)


def get_second(capture):
    if capture.isOpened():
        rate = capture.get(5)  # 帧速率
        FrameNumber = capture.get(7)  # 视频文件的帧数
        duration = FrameNumber / rate  # 帧速率/视频总帧数 是时间，除以60之后单位是分钟
        return int(rate), int(FrameNumber), int(duration)


def detect(video_path):
    video_name = video_path
    print(video_name)
    capture = cv2.VideoCapture(video_name)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fps = capture.get(cv2.CAP_PROP_FPS)  # 帧数
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高
    out = cv2.VideoWriter('../video_test/result.mp4', fourcc, fps, (width, height))  # 写入视频
    frame_count = 0
    fps_all = 0
    if capture.isOpened():
        while True:
            t1 = cv2.getTickCount()
            frame_count += 1
            print(f"第{frame_count} 帧", end=" ")
            ret, img = capture.read()
            if not ret:
                break
            dict_list = detect_Recognition_plate(detect_model, img, device, plate_rec_model, 640,
                                                 is_color=True)
            ori_img = draw_result(img, dict_list)
            t2 = cv2.getTickCount()
            infer_time = (t2 - t1) / cv2.getTickFrequency()
            fps = 1.0 / infer_time
            fps_all += fps
            str_fps = f'fps:{fps:.4f}'

            cv2.putText(ori_img, str_fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            out.write(ori_img)

    else:
        print("失败")

    capture.release()
    out.release()
    print(f"all frame is {frame_count},average fps is {fps_all / frame_count} fps")


class VideoBox(QWidget):
    VIDEO_TYPE_OFFLINE = 0
    VIDEO_TYPE_REAL_TIME = 1

    STATUS_INIT = 0
    STATUS_PLAYING = 1
    STATUS_PAUSE = 2

    def __init__(self, video_url="", video_type=0, auto_play=False):
        super().__init__()
        self.setWindowTitle("视频检测")

        icon_path = "../UI/car.png"
        self.setWindowIcon(QIcon(icon_path))

        self.video_processing_thread = None
        self.video_url = video_url
        self.video_type = video_type
        self.auto_play = auto_play
        self.status = self.STATUS_INIT

        # 固定的播放区域大小
        self.fixed_width = 800
        self.fixed_height = 600

        # 组件展示
        self.pictureLabel = QLabel()
        self.pictureLabel.setFixedSize(self.fixed_width, self.fixed_height)

        self.playButton = QPushButton()
        self.playButton.setEnabled(True)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.switch_video)

        self.selectButton = QPushButton("选择视频")
        self.selectButton.clicked.connect(self.select_video)

        # 创建开始检测按钮
        self.detectButton = QPushButton("开始检测")
        self.detectButton.clicked.connect(self.start_detection)

        self.progressSlider = QSlider(Qt.Horizontal)
        self.progressSlider.setMinimum(0)
        self.progressSlider.setMaximum(100)
        self.progressSlider.setValue(0)
        self.progressSlider.sliderMoved.connect(self.set_video_position)

        # 设置按钮的大小
        button_size = QSize(80, 40)
        self.playButton.setFixedSize(button_size)
        self.selectButton.setFixedSize(button_size)
        self.detectButton.setFixedSize(button_size)

        # 创建两个 QLabel 用于显示文字
        self.label1 = QLabel("")
        self.label2 = QLabel("")

        # 设置 QLabel 的样式
        font = QFont()
        font.setPointSize(12)
        self.label1.setFont(font)
        self.label2.setFont(font)

        # 创建垂直布局用于放置 QLabel
        info_layout = QVBoxLayout()
        info_layout.addWidget(self.label1)
        info_layout.addWidget(self.label2)

        # 创建水平布局放置按钮
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.selectButton)
        button_layout.addWidget(self.playButton)
        button_layout.addWidget(self.detectButton)

        # 创建垂直布局放置视频、播放条和按钮布局
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.pictureLabel, alignment=Qt.AlignLeft)
        video_layout.addWidget(self.progressSlider)
        video_layout.addLayout(button_layout)

        # 创建垂直布局放置 info_layout
        info_container = QWidget()
        info_container.setStyleSheet("background-color: white; border: 1px solid black;")
        info_container.setFixedSize(self.fixed_width // 3, self.fixed_height)
        info_container.setLayout(info_layout)

        # 创建水平布局用于放置视频、info_container
        main_layout = QHBoxLayout()
        main_layout.addLayout(video_layout)
        main_layout.addSpacing(20)
        main_layout.addWidget(info_container)

        # 设置整体布局
        self.setLayout(main_layout)

        # 设置窗口初始大小
        self.resize(self.fixed_width + self.fixed_width // 3, self.fixed_height + 50)

        # 设置按钮和播放条的绝对位置
        self.playButton.setGeometry(10, 10, button_size.width(), button_size.height())
        self.selectButton.setGeometry(60, 10, button_size.width(), button_size.height())
        self.progressSlider.setGeometry(10, self.fixed_height - 50, self.fixed_width - 20, 20)

        # timer 设置
        self.timer = VideoTimer()
        self.timer.timeSignal.signal[str].connect(self.show_video_images)

        # video 初始设置
        self.playCapture = VideoCapture()
        if self.video_url != "":
            self.set_timer_fps()
            if self.auto_play:
                self.switch_video()

    def start_detection(self):
        # 获取当前视频路径
        video_path = self.video_url
        if video_path == '':
            print('视频为空，无法检测！')
            return

        # 创建视频处理线程
        self.video_processing_thread = VideoProcessingThread(video_path)
        self.video_processing_thread.processing_finished.connect(self.processing_finished)

        # 启动视频处理线程
        self.video_processing_thread.start()

        # 更新label2
        self.update_label2_processing()

    def update_label2_processing(self):
        # 在视频处理线程开始时更新label2的文本
        self.label2.setText("视频处理中，请稍后...")

    def update_label2_finished(self):
        # 在视频处理线程结束时更新label2的文本
        video_path = self.video_url
        video_dir, video_file = os.path.split(video_path)
        save_file_name = "result.mp4"
        save_path = os.path.join(video_dir, save_file_name)
        save_path = save_path.replace('\\', '/')
        print(save_path)
        lines = [save_path[i:i + 23] for i in range(0, len(save_path), 23)]
        # 使用换行符连接子字符串
        formatted_save_path = '\n'.join(lines)
        self.label2.setText(f"视频处理完成！保存路径：\n{formatted_save_path}")

    def processing_finished(self):
        self.update_label2_finished()
        self.video_processing_thread.quit()
        self.video_processing_thread.wait()
        del self.video_processing_thread

    def reset(self):
        self.timer.stop()
        self.playCapture.release()
        self.status = VideoBox.STATUS_INIT
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def set_timer_fps(self):
        self.playCapture.open(self.video_url)
        fps = self.playCapture.get(CAP_PROP_FPS)
        self.timer.set_fps(fps)
        self.playCapture.release()

    def set_video(self, url, video_type=VIDEO_TYPE_OFFLINE, auto_play=False):
        self.reset()
        self.video_url = url
        self.video_type = video_type
        self.auto_play = auto_play
        self.set_timer_fps()
        if self.auto_play:
            self.switch_video()

    def play(self):
        if self.video_url == "" or self.video_url is None:
            return
        if not self.playCapture.isOpened():
            self.playCapture.open(self.video_url)
        self.timer.start()
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.status = VideoBox.STATUS_PLAYING

    def stop(self):
        if self.video_url == "" or self.video_url is None:
            return
        if self.playCapture.isOpened():
            self.timer.stop()
            if self.video_type is VideoBox.VIDEO_TYPE_REAL_TIME:
                self.playCapture.release()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.status = VideoBox.STATUS_PAUSE

    def re_play(self):
        if self.video_url == "" or self.video_url is None:
            return
        self.playCapture.release()
        self.playCapture.open(self.video_url)
        self.timer.start()
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.status = VideoBox.STATUS_PLAYING

    def set_video_position(self):
        if self.playCapture.isOpened():
            total_frames = int(self.playCapture.get(cv2.CAP_PROP_FRAME_COUNT))
            target_frame = int(self.progressSlider.value() / 100.0 * total_frames)
            self.playCapture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    def update_label1_text(self, text):
        lines = [text[i:i + 23] for i in range(0, len(text), 23)]
        # 使用换行符连接子字符串
        formatted_text = '\n'.join(lines)
        self.label1.setText(f"文件路径:\n{formatted_text}\n\n [点击播放键开始播放]")

    def show_video_images(self):
        if self.playCapture.isOpened():
            success, frame = self.playCapture.read()
            if success:
                # 颜色空间转换
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 获取原始帧的大小
                frame_height, frame_width, channel = rgb_frame.shape

                # 计算缩放比例
                scale_ratio = min(self.fixed_width / frame_width, self.fixed_height / frame_height)

                # 缩放帧
                resized_frame = cv2.resize(rgb_frame, (int(frame_width * scale_ratio), int(frame_height * scale_ratio)))

                # 将帧转换为显示格式
                height, width, channel = resized_frame.shape
                bytes_per_line = 3 * width
                qt_image = QImage(resized_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                qt_pixmap = QPixmap.fromImage(qt_image)

                # 设置 QLabel 的大小和显示缩放后的帧
                self.pictureLabel.setFixedSize(self.fixed_width, self.fixed_height)
                self.pictureLabel.setPixmap(qt_pixmap)

                current_frame = int(self.playCapture.get(cv2.CAP_PROP_POS_FRAMES))
                total_frames = int(self.playCapture.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    progress_value = int(current_frame / total_frames * 100)
                    self.progressSlider.setValue(progress_value)
            else:
                print("read failed, no frame data")
                success, frame = self.playCapture.read()
                if not success and self.video_type is VideoBox.VIDEO_TYPE_OFFLINE:
                    print("play finished")
                    self.reset()
                    self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
                return
        else:
            print("open file or capturing device error, init again")
            self.reset()

    def switch_video(self):
        if self.video_url == "" or self.video_url is None:
            return
        if self.status is VideoBox.STATUS_INIT:
            self.playCapture.open(self.video_url)
            self.timer.start()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        elif self.status is VideoBox.STATUS_PLAYING:
            self.timer.stop()
            if self.video_type is VideoBox.VIDEO_TYPE_REAL_TIME:
                self.playCapture.release()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        elif self.status is VideoBox.STATUS_PAUSE:
            if self.video_type is VideoBox.VIDEO_TYPE_REAL_TIME:
                self.playCapture.open(self.video_url)
            self.timer.start()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

        self.status = (VideoBox.STATUS_PLAYING,
                       VideoBox.STATUS_PAUSE,
                       VideoBox.STATUS_PLAYING)[self.status]

    def select_video(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mkv)")
        if file_path:
            self.set_video(file_path)
            self.update_label1_text(file_path)

    def closeEvent(self, event):
        # 关闭窗口时，调用 close() 关闭当前窗口
        self.close()


class VideoProcessingThread(QThread):
    processing_finished = pyqtSignal()

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        detect(self.video_path)
        self.processing_finished.emit()


class Communicate(QObject):
    signal = pyqtSignal(str)


class VideoTimer(QThread):

    def __init__(self, frequent=20):
        QThread.__init__(self)
        self.stopped = False
        self.frequent = frequent
        self.timeSignal = Communicate()
        self.mutex = QMutex()

    def run(self):
        with QMutexLocker(self.mutex):
            self.stopped = False
        while True:
            if self.stopped:
                return
            self.timeSignal.signal.emit("1")
            time.sleep(1 / self.frequent)

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def is_stopped(self):
        with QMutexLocker(self.mutex):
            return self.stopped

    def set_fps(self, fps):
        self.frequent = fps


if __name__ == "__main__":
    mapp = QApplication(sys.argv)
    mw = VideoBox()
    mw.show()
    sys.exit(mapp.exec_())
