import imaplib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import email
from email.header import decode_header
from plyer import notification
import time
import os
import re
import tkinter as tk
from tkinter import messagebox, filedialog

# 定义不同电子邮件提供商的服务器详细信息
EMAIL_SERVERS = {
    "qq.com": {"smtp": "smtp.qq.com", "imap": "imap.qq.com"},
    "gmail.com": {"smtp": "smtp.gmail.com", "imap": "imap.gmail.com"},
    "outlook.com": {"smtp": "smtp-mail.outlook.com", "imap": "imap-mail.outlook.com"},
    "yahoo.com": {"smtp": "smtp.mail.yahoo.com", "imap": "imap.mail.yahoo.com"},
    "icloud.com": {"smtp": "smtp.mail.me.com", "imap": "imap.mail.me.com"},
    "hotmail.com": {"smtp": "smtp.live.com", "imap": "imap-mail.outlook.com"},
    "aol.com": {"smtp": "smtp.aol.com", "imap": "imap.aol.com"},
    "zoho.com": {"smtp": "smtp.zoho.com", "imap": "imap.zoho.com"},
    "protonmail.com": {"smtp": "smtp.protonmail.com", "imap": "imap.protonmail.com"},
    "mail.com": {"smtp": "smtp.mail.com", "imap": "imap.mail.com"},
    "gmx.com": {"smtp": "smtp.gmx.com", "imap": "imap.gmx.com"},
    # 根据需要添加更多电子邮件提供商
}

SMTP_PORT = 587
IMAP_PORT = 993


def clean_text(text):
    """
    清理从邮件中提取的文本
    """
    return text.replace("\r\n", "\n").strip()


def get_email_content(msg):
    """
    提取邮件的正文内容，支持多部分邮件
    """
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            if content_type == "text/plain" and "attachment" not in content_disposition:
                return clean_text(part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8"))
            elif content_type == "text/html" and "attachment" not in content_disposition:
                return clean_text(part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8"))
    else:
        content_type = msg.get_content_type()
        if content_type == "text/plain":
            return clean_text(msg.get_payload(decode=True).decode(msg.get_content_charset() or "utf-8"))
        elif content_type == "text/html":
            return clean_text(msg.get_payload(decode=True).decode(msg.get_content_charset() or "utf-8"))
    return None


def sanitize_filename(filename, max_length=50):
    """
    清理文件名，删除或替换无效字符，并限制文件名长度。

    Args:
        filename (str): 原始文件名。
        max_length (int): 文件名最大长度（包括扩展名）。默认值为150。

    Returns:
        str: 处理后的安全文件名。
    """
    if not isinstance(filename, str):
        raise ValueError("Expected string or bytes-like object")

    # 替换无效字符为下划线
    filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
    # 删除换行符和制表符
    filename = re.sub(r'[\r\n\t]', "", filename)
    # 删除前后空格
    filename = filename.strip()

    # 如果文件名过长，进行截断处理
    if len(filename) > max_length:
        name, ext = re.match(r"^(.*?)(\.[^.]*)?$", filename).groups()
        ext = ext or ""  # 如果没有扩展名，设置为空字符串
        max_name_length = max_length - len(ext)  # 保留扩展名的长度

        # 截断文件名部分
        filename = name[:max_name_length] + ext

    return filename

def save_email(subject, content, file_path):
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 将邮件内容保存到文件
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(f"<h1>{subject}</h1>\n")
        file.write(content)


class EmailClient:
    def __init__(self):
        self.is_logged_in = False
        self.email_account = ""
        self.password = ""
        self.imap_connection = None
        self.smtp_connection = None

    def imap_login(self, account, password):
        # Try connecting to the IMAP server
        domain = account.split('@')[-1]
        if domain not in EMAIL_SERVERS:
            print("登录失败", f"不支持的邮件服务：{domain}")
            return 0
        imap_server = EMAIL_SERVERS[domain]["imap"]
        self.imap_connection = imaplib.IMAP4_SSL(imap_server, IMAP_PORT)
        self.imap_connection.login(account, password)  # Try login
        return 1
    
    def smtp_login(self, account, password):
        # Try connecting to the SMTP server
        domain = account.split('@')[-1]
        if domain not in EMAIL_SERVERS:
            print("登录失败", f"不支持的邮件服务：{domain}")
            return 0
        smtp_sever = EMAIL_SERVERS[domain]["smtp"]
        self.smtp_connection = smtplib.SMTP(smtp_sever, SMTP_PORT)
        self.smtp_connection.starttls()
        self.smtp_connection.login(account, password)
        return 1

    def validate_and_login(self, account, password):
        try:
            if not self.imap_login(account, password):
                return 0
            if not self.smtp_login(account, password):
                return 0
            
            self.is_logged_in = True
            self.email_account = account
            self.password = password
            print("Login Succeeded!")
            return 1
            
        except Exception as e:
            print("Login Failed!")
            self.imap_connection = None
            self.smtp_connection = None
            self.is_logged_in = False
            return 2
    
    def send_email(self, recipient, subject, body):
        if self.is_logged_in:
            # create a multibody message
            msg = MIMEMultipart()
            msg['From'] = self.email_account
            msg['To'] = recipient
            msg['Subject'] = subject

            # 将正文附加到消息实例
            # attach body to the message instance
            msg.attach(MIMEText(body, 'plain'))

            try:
                # send an email
                self.smtp_connection.sendmail(self.email_account, recipient, msg.as_string())

                print("sent succeeded!")
                print("SMTP relogin...")
                if self.smtp_login(self.email_account, self.password):
                    print("Relogin succeeded!")
                else:
                    print("Relogin failed!")
                    return 3
                return 0
                
            except Exception as e:
                print(f"sent failed: {e}")
                return 1
                
        else:
            print("Please login first!")
            return 2

    def receive_email(self, check_status='UNSEEN'):
        message_infos = []
        if self.is_logged_in:
            try:
                print("Searching emails...")
                # select inbox
                self.imap_connection.select("inbox")
                # search new emails
                status, messages = self.imap_connection.search(None, check_status)

                # process fresh emails
                for num in messages[0].split():
                    # get message contents
                    status, msg_data = self.imap_connection.fetch(num, "(RFC822)")
                    if status != "OK":
                        print("Error in processing email contents")
                        continue

                    for response_part in msg_data:
                        if isinstance(response_part, tuple):
                            msg = email.message_from_bytes(response_part[1])

                            # decode email subject
                            subject, encoding = decode_header(msg["Subject"])[0]
                            if isinstance(subject, bytes):
                                subject = subject.decode(encoding or "utf-8")
                            
                            # get contents of the email
                            content = get_email_content(msg)
                            sender = msg.get("From")
                            sender = sender.split('<')[1].split('>')[0]

                            # Prepare Message_Info data for BERT to process
                            message_info = {
                                'sender': sender,
                                'date': msg.get("Date"),
                                'subject': subject,
                                'content': content,
                            }
                            message_infos.append(message_info)

                            # get email status
                            email_status, data = self.imap_connection.fetch(num, "(FLAGS)")
                            if email_status != "OK":
                                print("Error in getting email status")
                                continue

                            # check whether is an unseen email
                            flags = data[0].decode()
                            if "\\Seen" not in flags:
                                # label the emails as seen
                                self.imap_connection.store(num, '+FLAGS', '\\Seen')
                print("Receive Success!")
                flag = 0
            except Exception as e:
                print(f"Receive failed: {e}")
                flag = 1
        else:
            print("Please login first!")
            flag = 2
        
        return message_infos, flag     # pass to BERT models

    def logout(self):
        if self.is_logged_in:
            if self.imap_connection:
                self.imap_connection.logout()
            if self.smtp_connection:
                self.smtp_connection.quit()

            # clear status
            self.is_logged_in = False
            self.email_account = ""
            # self.password = ""
            self.imap_connection = None
            self.smtp_connection = None
            print("Logout succeeded!")
        else:
            print("You did not login yet!")

    def notify(self, subject, content, sender):
        # send notification
        notification.notify(
            title=f"📧 来自{sender}的新邮件!",
            message=f"主题: {subject}\n主要内容: {content}",
            timeout=10
        )
    
            