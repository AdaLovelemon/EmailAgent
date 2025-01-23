import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import ImageTk, Image
import threading # 使用多线程而不是多进程因为目标是程序轻量化
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from model.generate import generate_response
from model.classify import EmailClassifier

from utils.email_utils import *

# 常量
IMAGE_PATH = "assets/background.jpg"
ICON_PATH = "assets/mail.ico"

# Path for saving the password
PASSWORD_FILE = "config/password.txt"




class MailApp:
    def __init__(self, root):
        self.root = root
        self.root.title("电子邮件客户端")
        self.root.geometry("800x400")
        self.root.resizable(False, False)

        # Set window icon
        self.root.iconbitmap(ICON_PATH)

        # Load the background image
        self.background_image = Image.open(IMAGE_PATH)

        # Create an Email Client
        self.email_client = EmailClient()

        # Create two canvases for different pages
        self.canvas = {'login': tk.Canvas(self.root), 'email': tk.Canvas(self.root)}

        for canvas in self.canvas.values():
            canvas.pack(fill='both', expand=True)
            canvas.config(cursor="@assets/cursor.cur")

        # Set up canvas switching and background update
        self.switch_canvas('email', 'login')
        # self.root.bind("<Configure>", lambda event: self.update_background(event))

        # Load saved password if available
        self.saved_username, self.saved_password = self.load_password()

        # Initialize Canvas 1 (Login Screen)
        self.setup_login_canvas()

        # Initialize Canvas 2 (Email functionality)
        self.setup_email_canvas()

        self.scanning_lock = False
        self.email_classifier = EmailClassifier()

        # self.executor = ThreadPoolExecutor(max_workers=4)


    # General Tools
    def switch_canvas(self, canvas_hide, canvas_show):
        self.canvas[canvas_hide].pack_forget()  # Hide canvas2
        self.canvas[canvas_show].pack(fill='both', expand=True)  # Show the selected canvas
        self.update_background(None, canvas_show)  # Trigger background update

    def update_background(self, event, canvas_name):
        if event is None:
            new_width = self.root.winfo_width()
            new_height = self.root.winfo_height()
        else:
            new_width = event.width
            new_height = event.height

        background_resized = self.background_image.resize((new_width, new_height))
        background_photo = ImageTk.PhotoImage(background_resized)

        self.canvas[canvas_name].create_image(0, 0, image=background_photo, anchor='nw')
        self.canvas[canvas_name].image = background_photo  # Keep a reference to avoid garbage collection

    def center_widget(self, widget, relx, rely):
        widget.place(relx=relx, rely=rely, anchor="center")


    # Email Canvas Tools
    def load_password(self):
        if os.path.exists(PASSWORD_FILE):
            with open(PASSWORD_FILE, "r") as f:
                username = f.readline().strip()
                password = f.readline().strip()
                return username, password
        return None, None
    
    def save_password(self, username, password):
        os.makedirs(os.path.dirname(PASSWORD_FILE), exist_ok=True)
        with open(PASSWORD_FILE, "w") as f:
            f.write(f"{username}\n{password}\n")

    def clear_account(self):
        if os.path.exists(PASSWORD_FILE):
            with open(PASSWORD_FILE, "w") as f:
                pass
        self.username_entry.delete(0, tk.END)
        self.password_entry.delete(0, tk.END)
        self.remember_check.deselect()
        
    def login_and_enter_canvas2(self):
        username = self.username_entry.get().strip()
        password = self.password_entry.get().strip()
        
        if username == "" or password == "":
            messagebox.showerror("登录失败", "请输入有效的账号和密码")
            return
        
        # Validate login with email server
        login_status = self.email_client.validate_and_login(username, password)
        if login_status == 0:
            messagebox.showerror("登录失败", f"不支持的邮件服务：{username.split('@')[-1]}")
            return
        elif login_status == 2:
            messagebox.showerror("登录失败", "用户名或密码错误")
            return

        # Entered logined status
        
        if self.remember_var.get():
            self.save_password(username, password)  # Save username and password if "Remember password" is checked
        messagebox.showinfo("登录成功", "您已成功登录！")
        
        # Pass username and password to second canvas and show it
        self.switch_canvas('login', 'email')


    def setup_login_canvas(self):
        # Canvas 1 content: Login screen
        self.username_label = tk.Label(self.canvas['login'], text="账号", bg="#f0f0f0")
        self.center_widget(self.username_label, 0.5, 0.25)

        self.username_entry = tk.Entry(self.canvas['login'], width=30)
        if self.saved_username:
            self.username_entry.insert(0, self.saved_username)
        self.center_widget(self.username_entry, 0.5, 0.3)

        self.password_label = tk.Label(self.canvas['login'], text="密码", bg="#f0f0f0")
        self.center_widget(self.password_label, 0.5, 0.35)

        self.password_entry = tk.Entry(self.canvas['login'], show="*", width=30)
        if self.saved_password:
            self.password_entry.insert(0, self.saved_password)
        self.center_widget(self.password_entry, 0.5, 0.4)

        self.remember_var = tk.IntVar()
        self.remember_check = tk.Checkbutton(self.canvas['login'], text="记住密码", variable=self.remember_var, bg="#f0f0f0")
        if self.saved_password:
            self.remember_check.select()
        else:
            self.remember_check.deselect()
        self.center_widget(self.remember_check, 0.5, 0.5)

        self.login_button = tk.Button(
            self.canvas['login'],
            text="登录",
            command=partial(self.login_and_enter_canvas2),
            font=("Arial", 14, "bold"),
            bg="#4CAF50",
            fg="white",
            relief="raised",
            bd=3,
            activebackground="#45a049",
            activeforeground="yellow"
        )
        self.center_widget(self.login_button, 0.5, 0.6)

        self.clear_account_button = tk.Button(
            self.canvas['login'],
            text="移除账户",
            command=self.clear_account,
            font=("Arial", 14, "bold"),
            bg="#66ccff",
            fg="white",
            relief="raised",
            bd=3,
            activebackground="#66cccc",
            activeforeground="yellow"
        )
        self.center_widget(self.clear_account_button, 0.5, 0.8)



    # Email Canvas Tools
    def toggle_scan(self):
        self.scan_active = not self.scan_active
        self.update_button_state(self.scan_button, self.scan_active)

        print(f"自动接收邮件: {self.scan_active}")
        if not self.scanning_lock:  # 互锁机制实现UI端控制扫描是否开启
            self.start_check_mail()

    def toggle_reply(self):
        self.reply_active = not self.reply_active
        self.update_button_state(self.reply_button, self.reply_active)
        print(f"自动回复邮件: {self.reply_active}")

    def browse_directory(self):
        directory = filedialog.askdirectory()  # Use file dialog to browse directory
        if directory:
            self.save_dir_entry.delete(0, tk.END)
            self.save_dir_entry.insert(0, directory)

    def logout_and_back_canvas1(self):
        self.email_client.logout()
        self.switch_canvas('email', 'login')
        messagebox.showinfo("登出成功", "您已成功登出!")

    def update_button_state(self, button, active):
        if active:
            button.config(bg="green")
        else:
            button.config(bg="#66ccff")

    def send_email(self, recipient, subject, body):
        # Logic to send email
        status = self.email_client.send_email(recipient, subject, body)
        if status == 0:
            pass
            # messagebox.showinfo("邮件发送", "邮件发送成功")
        elif status == 1:
            messagebox.showerror("邮件发送", "邮件发送失败")
        elif status == 2:
            messagebox.showerror("邮件发送", "请先登录账号")
        elif status == 3:
            messagebox.showerror("邮箱登录", "邮箱登录异常")

    def start_send_email(self):
        recipient = self.recipient_entry.get().strip()
        subject = self.subject_entry.get().strip()
        body = self.body_entry.get("1.0", tk.END).strip()
        self.send_email(recipient, subject, body)

    def start_check_mail(self):
        # Logic to check for new emails
        self.scanning_lock = True
        self.scan_thread = threading.Thread(target=self.check_mail)
        self.scan_thread.daemon = True
        self.scan_thread.start()   
        # 但是线程如何join是需要考虑的

    def check_mail(self, check_status='UNSEEN'):
        # 还是需要多线程，主线程不能有while循环
        while self.scan_active:
            print('scanning...')
            contents, status = self.email_client.receive_email(check_status=check_status)
            if len(contents) != 0:
                # Classify
                check_results, categories = self.auto_classify(contents)
                for i in check_results:
                    msg = contents[i]
                    sender = msg['sender']
                    subject = msg['subject']
                    content = msg['content']
                    self.notify(subject, content, sender)

                    if self.reply_active:
                        # Generate Reply
                        # TODO
                        category = categories[i]
                        self.auto_reply(sender, msg, category)    # 使用多线程同时回复会被视为攻击
                        time.sleep(2)
            time.sleep(10)

        self.scanning_lock = False


    def notify(self, subject, content, sender):
        # send notification
        notification.notify(
            title=f"📧 来自{sender}的新邮件!",
            message=f"主题: {subject}\n主要内容: {content}",
            timeout=10
        )
    
    
    def auto_reply(self, sender, content, category):
        re_subject, re_body = self.auto_generate(category, content)
        print(re_body)

        # TODO
        self.send_email(sender, re_subject, re_body)


    def setup_email_canvas(self):
        # Canvas 2 content: Email functionality
        self.recipient_label = tk.Label(self.canvas['email'], text="收件人", bg="#f0f0f0")
        self.center_widget(self.recipient_label, 0.5, 0.1)
        self.recipient_entry = tk.Entry(self.canvas['email'], width=30)
        self.center_widget(self.recipient_entry, 0.5, 0.15)

        self.subject_label = tk.Label(self.canvas['email'], text="主题", bg="#f0f0f0")
        self.center_widget(self.subject_label, 0.5, 0.2)
        self.subject_entry = tk.Entry(self.canvas['email'], width=30)
        self.center_widget(self.subject_entry, 0.5, 0.25)

        self.body_label = tk.Label(self.canvas['email'], text="正文", bg="#f0f0f0")
        self.center_widget(self.body_label, 0.5, 0.3)
        self.body_entry = tk.Text(self.canvas['email'], width=30, height=6)
        self.center_widget(self.body_entry, 0.5, 0.45)

        self.scan_active = False
        self.scan_button = tk.Button(
            self.canvas['email'],
            text="自动接收邮件",
            command=self.toggle_scan,
            bg="#66ccff",  # 背景颜色
            fg="white",  # 前景颜色
            font=("Arial", 12, "bold"),  # 字体
            activebackground="#66cccc",  # 按下时的背景颜色
            activeforeground="yellow",  # 按下时的前景颜色
            highlightthickness=2,  # 高亮边框的厚度
            bd=3  # 边框宽度
        )
        self.center_widget(self.scan_button, 0.8, 0.2)

        self.reply_active = False
        self.reply_button = tk.Button(
            self.canvas['email'],
            text="自动回复邮件",
            command=self.toggle_reply,
            bg="#66ccff",  # 背景颜色
            fg="white",  # 前景颜色
            font=("Arial", 12, "bold"),  # 字体
            activebackground="#66cccc",  # 按下时的背景颜色
            activeforeground="yellow",  # 按下时的前景颜色
            highlightthickness=2,  # 高亮边框的厚度
            bd=3  # 边框宽度
        )
        self.center_widget(self.reply_button, 0.8, 0.4)

        self.send_button = tk.Button(
            self.canvas['email'],
            text="发送邮件",
            command=partial(self.start_send_email),     # 读取信息需要函数触发
            font=("Arial", 14, "bold"),
            bg="#4CAF50",
            fg="white",
            relief="raised",
            bd=3,
            activebackground="#45a049",
            activeforeground="yellow"
        )
        self.center_widget(self.send_button, 0.5, 0.65)

        self.switch_to_canvas1_button = tk.Button(self.canvas['email'], text="返回登录界面", command=partial(self.logout_and_back_canvas1))
        self.center_widget(self.switch_to_canvas1_button, 0.7, 0.9)



    # TODO
    def auto_classify(self, contents):
        # Classify
        # 给定一列表的dict，返回重要邮件的序号（或Ture, False）序列
        categories, results = [], []
        for i, content in enumerate(contents):
            category = self.email_classifier.classify(content['subject'], content['content'])
            print(category)
            if category != 'spam':
                results.append(i)
                category = category.replace('_', ' ')
                categories.append(category)
            
        return results, categories
    
    def auto_generate(self, category, content):
        subject = content['subject']
        body = content['content']
        body = body.strip()
        re_body = generate_response(body, category)
        re_subject = "Re: " + subject

        return re_subject, re_body
    



    # mainloop
    def run(self):
        # Start the Tkinter main loop
        self.root.mainloop()
