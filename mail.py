import tkinter as tk
from tkinter import messagebox
from functools import partial
from PIL import Image, ImageTk
import os
import imaplib

from utils.UI_utils import *


if __name__ == "__main__":
    root = tk.Tk()
    app = MailApp(root)
    app.run()
