import base64
from io import BytesIO
from PIL import Image
import pyautogui
import sqlite3
import smtplib
from email.mime.text import MIMEText

from api_key import email_id, email_pwd

def encode_image(image):
    buffered = BytesIO()
    w,h = image.size
    if w>512 or h>512:
        scale = 512 / max(w,h)
    else:
        scale = 1.0
    resize_im = image.resize((int(w*scale),int(h*scale))).convert('RGB')
    resize_im.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def get_indoor_image():
    return pyautogui.screenshot(region=(224,129,702-224,488-129))
def get_outdoor_image():
    return pyautogui.screenshot(region=(224,489,702-224,488-129))

def init_db():
    conn = sqlite3.connect("database/db.db")
    cur = conn.cursor()
    cur.execute("CREATE TABLE events(id INTEGER PRIMARY KEY AUTOINCREMENT, place VARCHAR(10), description VARCHAR(255))")
    conn.close()

def save_sqlite3(place, description):
    conn = sqlite3.connect("database/db.db")
    cur = conn.cursor()
    cur.execute("INSERT INTO events(place, description) VALUES ('"+place+"','"+description+"')")
    conn.commit()
    conn.close()

def get_db_msg(place):
    conn = sqlite3.connect("database/db.db")
    cur = conn.cursor()
    cur.execute("SELECT description FROM events WHERE place='"+place+"' ORDER BY id DESC LIMIT 1;")
    rows = cur.fetchall()
    conn.close()
    if len(rows) == 0:
        return ""
    return rows[0]

def send_email(subject, content):
    msg = MIMEText(content)
    msg['Subject'] = subject
    msg['From'] = email_id+"@gmail.com"
    msg['To'] = email_id+"@gmail.com"
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login(email_id, email_pwd)
    s.sendmail(email_id+"@gmail.com", email_id+"@gmail.com", msg.as_string())
    s.quit()
