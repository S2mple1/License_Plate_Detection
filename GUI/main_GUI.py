import tkinter as tk
from tkinter import Text, Frame, Toplevel
from PIL import Image, ImageTk
from GUI import LicensePlateDetectorGUI
from V_GUI import *


def video_detection():
    mapp = QApplication(sys.argv)
    window = VideoBox()
    window.show()
    mapp.exec_()


class Detect_main:
    def __init__(self, root, width, height):
        self.image_tk = None
        self.root = root
        self.root.title("车牌识别系统")
        self.root.geometry("%dx%d+%d+%d" % (width, height, 200, 50))
        icon_path = "../UI/car.ico"
        self.root.iconbitmap(default=icon_path)

        # 左侧图片区域
        self.image_label = tk.Label(root, text='pic')
        self.image_label.place(x=0, y=0, width=625, height=600)
        # 设置图片路径
        image_path = "../UI/background.jpg"  # 请替换为实际的图片路径
        self.set_image(image_path, width=625, height=600)  # 设置 label 的大小为 600x600

        # 图片检测按钮
        image_button = tk.Button(self.root, text="图片识别", command=self.image_detection, font=('微软雅黑', 12))
        image_button.place(x=662.5, y=125, width=100, height=75)

        # 视频检测按钮
        video_button = tk.Button(self.root, text="视频识别", command=video_detection, font=('微软雅黑', 12))
        video_button.place(x=662.5, y=275, width=100, height=75)

        # 开发信息按钮
        info_button = tk.Button(self.root, text="开发信息", command=self.staff_info, font=('微软雅黑', 12))
        info_button.place(x=662.5, y=425, width=100, height=75)

    def image_detection(self):
        image_detection_window = Toplevel(self.root)
        LicensePlateDetectorGUI(image_detection_window, 1200, 700)

    def staff_info(self):
        info_window = Toplevel(self.root)
        info_window.title("开发信息")
        info_window.geometry("%dx%d+%d+%d" % (500, 300, 300, 200))

        frame = Frame(info_window, padx=10, pady=10)
        frame.pack(expand=True, fill='both')

        info_text = (
            "\n\n开发人员：周豪捷、刘伯钰——北京理工大学2021级\n\n"
            "联系邮箱：midkingggg@gmail.com\n\n"
            "详细使用说明请看README.md"
        )
        # 使用 Text 组件
        info_text_widget = Text(frame, wrap='word', font=('微软雅黑', 14), height=10, width=40)
        info_text_widget.insert(tk.END, info_text)
        info_text_widget.pack(expand=True, fill='both')
        # 禁止编辑
        info_text_widget.config(state=tk.DISABLED)

    def set_image(self, image_path, width, height):
        original_image = Image.open(image_path)
        # 将图片调整为 label 的大小
        resized_image = original_image.resize((width, height), Image.LANCZOS)
        # 将图片转换为 Tkinter PhotoImage 对象
        self.image_tk = ImageTk.PhotoImage(resized_image)
        # 在 Label 中显示图片
        self.image_label.config(image=self.image_tk, width=width, height=height)
        self.image_label.image = self.image_tk


if __name__ == "__main__":
    root = tk.Tk()
    app = Detect_main(root, 800, 600)
    root.mainloop()
