import os
import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename
import cv2
from PIL import Image, ImageTk
from PIL.Image import Resampling
from detect_plate import detect_Recognition_plate, draw_result, init_model, load_model, get_second
import torch


class LicensePlateDetectorGUI:
    def __init__(self, root, width, height):
        self.result_img_tk = None
        self.original_img_tk = None
        self.root = root
        self.width = width
        self.height = height
        self.root.geometry("%dx%d+%d+%d" % (width, height, 200, 50))
        self.root.title("车牌识别")
        icon_path = "../UI/car.ico"
        self.root.iconbitmap(default=icon_path)

        # 初始化模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detect_model = load_model('../weights/plate_rec.pt', self.device)
        self.plate_rec_model = init_model(self.device, '../weights/plate_rec_color.pth', is_color=True)

        # 初始化其他变量
        self.image_path = ''

        # 创建标签
        Label(self.root, text='原图:', font=('微软雅黑', 18)).place(x=48, y=10)
        Label(self.root, text='识别结果:', font=('微软雅黑', 18)).place(x=640, y=10)

        # 创建用于显示原图的 Canvas
        self.original_canvas = Canvas(self.root, width=512, height=512, bg='white', relief='solid', borderwidth=1)
        self.original_canvas.place(x=48, y=50)

        # 创建用于显示检测结果的 Canvas
        self.result_canvas = Canvas(self.root, width=512, height=512, bg='white', relief='solid', borderwidth=1)
        self.result_canvas.place(x=640, y=50)

        # 创建选择图片的按钮
        self.select_image_button = tk.Button(self.root, text='选择图片', command=self.select_image, font=('微软雅黑', 12))
        self.select_image_button.place(x=300, y=600, width=100, height=50)

        # 创建开始检测的按钮
        self.detect_button = tk.Button(self.root, text='开始识别', command=self.detect_license_plate, font=('微软雅黑', 12))
        self.detect_button.place(x=550, y=600, width=100, height=50)

        # 创建清空按钮
        self.clear_button = tk.Button(self.root, text='清空图片', command=self.clear, font=('微软雅黑', 12))
        self.clear_button.place(x=800, y=600, width=100, height=50)
        print("已启动！开始识别！")

    def select_image(self):
        sv = StringVar()
        sv.set(askopenfilename(title="选择图片文件",
                               filetypes=[("Images", "*.png;*.xpm;*.jpg;*.bmp"),
                                          ("All Files", "*.*")]))
        self.image_path = Entry(self.root, state='readonly', textvariable=sv).get()
        print(self.image_path)
        self.original_canvas.delete('all')
        self.result_canvas.delete('all')
        self.show_selected_image()

    def detect_license_plate(self):
        if not self.image_path:
            print("请先选择图片文件")
            return

        # 加载图像并进行检测
        img = cv2.imread(self.image_path)
        dict_list = detect_Recognition_plate(self.detect_model, img, self.device, self.plate_rec_model, 640,
                                             is_color=True)
        result_img_save = draw_result(img, dict_list)
        result_img_rgb = cv2.cvtColor(result_img_save, cv2.COLOR_BGR2RGB)
        result_img = Image.fromarray(result_img_rgb)
        # 显示原图和检测结果图像
        self.show_result_images(result_img)
        save_path = '../imgs_test/img_test_result'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        img_name = os.path.basename(self.image_path)
        save_img_path = os.path.join(save_path, img_name)
        cv2.imwrite(save_img_path, result_img_save)

    def show_selected_image(self):
        if self.image_path:
            img_open = Image.open(self.image_path)
            img_open = img_open.resize((512, 512), Resampling.LANCZOS)

            self.original_img_tk = ImageTk.PhotoImage(img_open)
            self.original_canvas.create_image(258, 258, image=self.original_img_tk, anchor='center')

    def show_result_images(self, result_img):
        result_img = result_img.resize((512, 512), Resampling.LANCZOS)
        self.result_img_tk = ImageTk.PhotoImage(result_img)

        self.result_canvas.create_image(258, 258, image=self.result_img_tk, anchor='center')

    def clear(self):
        self.original_canvas.delete('all')
        self.result_canvas.delete('all')
        self.image_path = None


if __name__ == "__main__":
    print("启动中...请稍后...")
    root = tk.Tk()
    width = 1200
    height = 700
    gui = LicensePlateDetectorGUI(root, width, height)
    root.mainloop()
