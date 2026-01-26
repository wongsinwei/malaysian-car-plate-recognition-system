import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import datetime
import os
import sys
import csv
import re
import numpy as np
from collections import Counter
from ultralytics import YOLO
import easyocr
import threading

# --- FIREBASE IMPORT ---
# Ensure final_system_segmentation.py is in the same folder
try:
    from final_system_segmentation import ref, db_manager
except ImportError:
    # print("Firebase/Config not found. Cloud features disabled.")
    ref = None
    db_manager = None

# --- GLOBAL UI CONFIG ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# UI Constants for unification
FONT_HEADER = ("Roboto", 24, "bold")
FONT_SUBHEADER = ("Roboto", 18, "bold")
FONT_BODY = ("Roboto", 14)
FONT_BOLD = ("Roboto", 14, "bold")
COLOR_ACCENT = "#1f6aa5"  # Standard Blue
COLOR_SUCCESS = "#2cc985" # Green
COLOR_WARNING = "#e2983d" # Orange
COLOR_DANGER = "#c92c2c"  # Red
COLOR_CARD = "#2b2b2b"    # Card Background

MALAYSIA_PLATE_REGEX = re.compile(r'^([A-Z]{1,3})(\d{1,4})([A-Z]?)$')

VANITY_PREFIXES = [
    "PUTRAJAYA", "PROTON", "PERODUA", "WAJA", "SUKOM", "LIMO", "RIMAU",
    "BAMBEE", "IM4U", "1M4U", "PATRIOT", "VIP", "VIPS", "PERFECT", "NAAM",
    "G1M", "GP", "US", "UP", "A1M", "GOLD", "MALAYSIA", "NBOS", "GTR",
    "SAM", "K1M", "T1M", "FFF", "GG", "G", "FD", "FE", "FB", "X", "XX",
    "YY", "UU", "Q", "KRISS", "LOTUS", "MADANI", "NBOS", "PETRA", "PUTRA",
    "PERSONA", "PERDANA", "SATRIA", "SAS", "TIARA", "UNIMAS", "UNISZA", "UTEM",
    "UiTM", "IIUM", "WAJA", "WCEC", "XIIINAM", "XOIC", 'XXVIASEAN', "XXXIDB",
    "UUU"
]

# ==========================================
# GLOBAL SETTINGS MANAGER
# ==========================================
class SystemConfig:
    KNOWN_WIDTH = 1.8  # Avg car width in meters
    FOCAL_LENGTH = 500 # Default calibration
    TRIGGER_LINE_RATIO = 0.75 # Position of line (0.75 = 75% down)
    CONFIDENCE_THRESHOLD = 0.50
    LINE_OPACITY = 0.5

    @classmethod
    def get_trigger_y(cls, frame_height):
        return int(frame_height * cls.TRIGGER_LINE_RATIO)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def estimate_distance_and_size(box_width, box_height):
    if box_width == 0: return 0.0, 0.0
    distance_meters = (SystemConfig.KNOWN_WIDTH * SystemConfig.FOCAL_LENGTH) / box_width
    real_height_meters = (box_height * distance_meters) / SystemConfig.FOCAL_LENGTH
    return distance_meters, real_height_meters

def preprocess_plate(img):
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    return sharpened

def auto_correct_plate(text):
    for vp in VANITY_PREFIXES:
        if text.startswith(vp):
            return text
    # Prefix corrections: Common letter misreads (add more based on your logs)
    prefix_corrections = {'O': 'Q', 'C': 'C', 'D': 'D', 'G': 'G', 'N':'W'}  # e.g., 'O' often 'Q' in prefixes
    # Suffix corrections: Same as before, digit-focused
    suffix_corrections = {'B': '8', 'O': '0', 'D': '0', 'I': '1', 'S': '5', 'Z': '2', 'Q': '0', 'G': '6', 'J':'3','Z':'7'}
    
    text = text.upper().replace(" ", "").replace("-", "")
    if len(text) < 2: return text
    
    if MALAYSIA_PLATE_REGEX.match(text):
        return text

    # Find numeric start (more robust: look for first sequence of 1+ digits)
    match = re.search(r'\d+', text)
    if not match:
        return text  # Cannot recover
    prefix = text[:match.start()]
    rest   = text[match.start():]

    # Prefix: letters only, max 3
    prefix = ''.join(prefix_corrections.get(c,c) for c in prefix if c.isalpha())[:3]

    # Extract numeric
    digits = ''.join(suffix_corrections.get(c,c) for c in rest if c.isalnum())
    number = ''.join(c for c in digits if c.isdigit())[:4]

    # Suffix: 1 letter max
    suffix_letters = ''.join(c for c in rest if c.isalpha())
    suffix = suffix_letters[-1] if suffix_letters else ''

    candidate = prefix + number + suffix

    # Final validation
    if MALAYSIA_PLATE_REGEX.match(candidate):
        return candidate

    return text  # fallback

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # PyInstaller temp folder
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ==========================================
# AUTHENTICATION FRAMES
# ==========================================
class RegisterFrame(ctk.CTkFrame):
    def __init__(self, master, on_back):
        super().__init__(master, fg_color="transparent")
        self.place(relx=0.5, rely=0.5, anchor="center")
        self.on_back = on_back

        # Card Container
        self.card = ctk.CTkFrame(self, width=400, corner_radius=15, fg_color=COLOR_CARD)
        self.card.pack(padx=20, pady=20, fill="both")

        ctk.CTkLabel(self.card, text="Create Account", font=FONT_HEADER).pack(pady=(30, 20))

        self.entry_user = ctk.CTkEntry(self.card, placeholder_text="Username", width=280, height=40, font=FONT_BODY)
        self.entry_user.pack(pady=10)
        
        self.entry_email = ctk.CTkEntry(self.card, placeholder_text="Email", width=280, height=40, font=FONT_BODY)
        self.entry_email.pack(pady=10)
        
        self.entry_pass = ctk.CTkEntry(self.card, placeholder_text="Password", show="*", width=280, height=40, font=FONT_BODY)
        self.entry_pass.pack(pady=10)

        self.lbl_status = ctk.CTkLabel(self.card, text="", font=("Roboto", 12), text_color=COLOR_DANGER)
        self.lbl_status.pack(pady=(5, 5))

        ctk.CTkButton(self.card, text="REGISTER", fg_color=COLOR_SUCCESS, hover_color="#26ad73", width=280, height=40, font=FONT_BOLD, command=self.do_register).pack(pady=(20, 10))
        ctk.CTkButton(self.card, text="Back to Login", fg_color="transparent", text_color="#aaa", hover_color="#333", width=280, command=on_back).pack(pady=(0, 30))

    def validate_email(self, email):
        # Strict Regex for Email Format
        pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        return re.match(pattern, email) is not None

    def do_register(self):
        u = self.entry_user.get().strip()
        e = self.entry_email.get().strip()
        p = self.entry_pass.get().strip()

        # 1. Basic Empty Check
        if not u or not e or not p:
            self.lbl_status.configure(text="All fields are required!", text_color=COLOR_DANGER)
            return

        # 2. Email Format Check
        if not self.validate_email(e):
            self.lbl_status.configure(text="Invalid email format!", text_color=COLOR_DANGER)
            return

        # 4. Attempt Registration
        # We assume db_manager.register_user returns (True, "Success") or (False, "Error Message")
        success, msg = db_manager.register_user(u, p, e)
        
        if success:
            self.lbl_status.configure(text="Success! Redirecting...", text_color=COLOR_SUCCESS)
            self.after(500, self.on_back) # Wait 1 second then go back
        else:
            # Show the specific error from Firebase (e.g., "Email already exists")
            self.lbl_status.configure(text=str(msg), text_color=COLOR_DANGER)

# ==========================================
# PAGE: LOGIN
# ==========================================
class LoginFrame(ctk.CTkFrame):
    def __init__(self, master, on_success, on_register):
        super().__init__(master, fg_color="transparent")
        self.place(relx=0.5, rely=0.5, anchor="center")
        self.on_success = on_success
        self.on_register = on_register

        # Card Container
        self.card = ctk.CTkFrame(self, width=400, corner_radius=15, fg_color=COLOR_CARD)
        self.card.pack(padx=20, pady=20, fill="both")

        ctk.CTkLabel(self.card, text="LPR System", font=FONT_HEADER).pack(pady=(30, 5))
        
        self.role_var = ctk.StringVar(value="User")
        ctk.CTkSwitch(self.card, text="Login as Admin", variable=self.role_var, onvalue="Admin", offvalue="User", font=FONT_BOLD).pack(pady=10)

        self.entry_user = ctk.CTkEntry(self.card, placeholder_text="Username", width=280, height=40, font=FONT_BODY)
        self.entry_user.pack(pady=10)
        
        self.entry_pass = ctk.CTkEntry(self.card, placeholder_text="Password", show="*", width=280, height=40, font=FONT_BODY)
        self.entry_pass.pack(pady=10)
        
        self.lbl_status = ctk.CTkLabel(self.card, text="", font=("Roboto", 12), text_color=COLOR_DANGER)
        self.lbl_status.pack(pady=(5, 5))

        self.entry_pass.bind('<Return>', lambda e: self.attempt_login())

        ctk.CTkButton(self.card, text="LOGIN", fg_color=COLOR_ACCENT, width=280, height=40, font=FONT_BOLD, command=self.attempt_login).pack(pady=(20, 10))
        ctk.CTkButton(self.card, text="Create Account", fg_color="transparent", text_color="#aaa", hover_color="#333", width=280, command=on_register).pack(pady=(0, 30))

    def attempt_login(self):
        u = self.entry_user.get()
        p = self.entry_pass.get()
        role = self.role_var.get()

        self.lbl_status.configure(text="")

        if role == "Admin":
            user_data = db_manager.login_admin(u, p) 
            if user_data:
                self.on_success("ADMIN", user_data)
            else:
                self.lbl_status.configure(text="Invalid Admin Credentials", text_color=COLOR_DANGER)
        else:
            user_data = db_manager.login_user(u, p)
            if user_data:
                self.on_success("USER", user_data)
            else:
                self.lbl_status.configure(text="Invalid Username or Password", text_color=COLOR_DANGER)

class CameraSelectionFrame(ctk.CTkFrame):
    def __init__(self, master, user_data, on_launch, on_logout):
        super().__init__(master)
        self.pack(fill="both", expand=True, padx=40, pady=40)
        self.user_id, self.username = user_data
        self.on_launch = on_launch
        self.on_logout = on_logout

        # Top Bar
        top_bar = ctk.CTkFrame(self, fg_color="transparent")
        top_bar.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(top_bar, text=f"Welcome, {self.username}", font=FONT_HEADER).pack(side="left")
        
        action_frame = ctk.CTkFrame(top_bar, fg_color="transparent")
        action_frame.pack(side="right")
        
        ctk.CTkButton(action_frame, text="My Profile", fg_color=COLOR_WARNING, text_color="black", width=120, height=35,
                      command=self.edit_own_profile).pack(side="left", padx=10)
        ctk.CTkButton(action_frame, text="Logout", fg_color=COLOR_DANGER, width=100, height=35, 
                      command=on_logout).pack(side="left")

        # Content Area
        content = ctk.CTkFrame(self, fg_color=COLOR_CARD, corner_radius=15)
        content.pack(fill="both", expand=True, padx=0, pady=10)

        # Add Camera Section
        add_frame = ctk.CTkFrame(content, fg_color="transparent")
        add_frame.pack(fill="x", padx=20, pady=20)
        
        self.entry_ip = ctk.CTkEntry(add_frame, placeholder_text="IP Address or '0' for Webcam", height=40, font=FONT_BODY)
        self.entry_ip.pack(side="left", fill="x", expand=True, padx=(0, 10))
        ctk.CTkButton(add_frame, text="+ Add Camera", fg_color=COLOR_SUCCESS, height=40, font=FONT_BOLD, command=self.add_cam).pack(side="right")

        ctk.CTkLabel(content, text="Connected Cameras", font=FONT_SUBHEADER, text_color="gray").pack(anchor="w", padx=20, pady=(10, 5))

        # List
        self.scroll = ctk.CTkScrollableFrame(content, fg_color="transparent")
        self.scroll.pack(fill="both", expand=True, padx=10, pady=10)
        self.load_cameras()

    def add_cam(self):
        ip = self.entry_ip.get()
        if ip:
            db_manager.add_camera(self.user_id, ip)
            self.load_cameras()
            self.entry_ip.delete(0, 'end')

    def load_cameras(self):
        for w in self.scroll.winfo_children(): w.destroy()
        cams = db_manager.get_user_cameras(self.user_id)

        if not cams:
            ctk.CTkLabel(self.scroll, text="No cameras found. Add one above.", text_color="gray").pack(pady=40)
        
        for cid, ip in cams:
            row = ctk.CTkFrame(self.scroll, fg_color="#3a3a3a", corner_radius=10)
            row.pack(fill="x", pady=5, padx=5)

            ctk.CTkLabel(row, text=f"📹  {ip}", font=FONT_BOLD).pack(side="left", padx=20, pady=15)
            
            ctk.CTkButton(row, text="DEL", width=50, height=30, fg_color=COLOR_DANGER,
                         command=lambda c=cid: self.delete_cam(c)).pack(side="right", padx=(5, 15))
            
            ctk.CTkButton(row, text="LOGS", width=80, height=30, fg_color="#555",
                         command=lambda s=ip: self.open_history(s)).pack(side="right", padx=5)
            
            ctk.CTkButton(row, text="LAUNCH", width=100, height=30, fg_color=COLOR_ACCENT,
                         command=lambda s=ip: self.on_launch(s)).pack(side="right", padx=5)

    def delete_cam(self, cid):
        db_manager.delete_camera(self.user_id, cid)
        self.load_cameras()

    def open_history(self, filter_ip):
        UserHistoryWindow(self, self.user_id, filter_source=filter_ip)

    def edit_own_profile(self):
        try:
            user_data = ref.child('users').child(self.user_id).get()
            current_email = user_data.get('email', '')
            EditUserWindow(self, self.user_id, self.username, current_email, self.refresh_welcome)
        except Exception as e:
            pass # print(f"Error fetching profile: {e}")

    def refresh_welcome(self):
        u_data = ref.child('users').child(self.user_id).get()
        self.username = u_data.get('username')
        pass

# ==========================================
# POPUP: SETTINGS WINDOW
# ==========================================
class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__(master)
        self.attributes('-topmost', True)
        self.title("System Configuration")
        self.geometry("600x500")
        
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(container, text="Detection Settings", font=FONT_SUBHEADER).pack(pady=(0, 20))

        # --- 1. Trigger Line ---
        self.create_slider_group(container, "Trigger Line Position", 0.1, 0.9, SystemConfig.TRIGGER_LINE_RATIO, 
                                 lambda v: self.update_config("TRIGGER_LINE_RATIO", v), "slider_line", "lbl_line")

        # --- 2. Opacity ---
        self.create_slider_group(container, "Line Opacity", 0.0, 1.0, SystemConfig.LINE_OPACITY,
                                 lambda v: self.update_config("LINE_OPACITY", v), "slider_opacity", "lbl_opacity")

        # --- 3. Confidence ---
        self.create_slider_group(container, "AI Confidence Threshold", 0.3, 0.95, SystemConfig.CONFIDENCE_THRESHOLD,
                                 lambda v: self.update_config("CONFIDENCE_THRESHOLD", v), "slider_conf", "lbl_conf")
        
        # --- Focal Length ---
        ctk.CTkLabel(container, text="Focal Length (Calibration)", font=FONT_BOLD).pack(anchor="w", pady=(15, 5))
        self.entry_focal = ctk.CTkEntry(container)
        self.entry_focal.insert(0, str(SystemConfig.FOCAL_LENGTH))
        self.entry_focal.pack(fill="x")

        ctk.CTkButton(container, text="Save & Close", fg_color=COLOR_SUCCESS, height=40, font=FONT_BOLD, command=self.save_and_close).pack(pady=30)

    def create_slider_group(self, parent, title, min_val, max_val, current, command, slider_attr, label_attr):
        ctk.CTkLabel(parent, text=title, font=FONT_BOLD).pack(anchor="w", pady=(10,0))
        
        val_lbl = ctk.CTkLabel(parent, text=f"{current:.2f}", text_color=COLOR_WARNING, font=FONT_BOLD)
        val_lbl.pack(anchor="e", pady=(0,0))
        setattr(self, label_attr, val_lbl)

        slider = ctk.CTkSlider(parent, from_=min_val, to=max_val, command=command)
        slider.set(current)
        slider.pack(fill="x", pady=5)
        setattr(self, slider_attr, slider)

    def update_config(self, key, value):
        setattr(SystemConfig, key, value)
        # Update label
        lbl_map = {
            "TRIGGER_LINE_RATIO": self.lbl_line,
            "LINE_OPACITY": self.lbl_opacity,
            "CONFIDENCE_THRESHOLD": self.lbl_conf
        }
        if key in lbl_map:
            lbl_map[key].configure(text=f"{value:.2f}")

    def save_and_close(self):
        try:
            val = float(self.entry_focal.get())
            SystemConfig.FOCAL_LENGTH = val
        except ValueError: pass
        self.destroy()

# ==========================================
# POPUP: EDIT RECORD WINDOW (ADMIN)
# ==========================================
class EditRecordWindow(ctk.CTkToplevel):
    def __init__(self, master, user_id, plate, record, on_save_callback):
        super().__init__(master)
        self.attributes('-topmost', True)
        self.title(f"Edit Record: {plate}")
        self.geometry("450x600")
        self.user_id = user_id
        self.old_plate = plate
        self.record = record
        self.on_save = on_save_callback

        self.grab_set()
        self.focus_force()

        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(container, text="Edit Database Record", font=FONT_SUBHEADER).pack(pady=(0, 20))

        self.entry_plate = self.create_field(container, "Plate Number:", plate)
        self.entry_color = self.create_field(container, "Color:", record.get('color', ''))
        self.entry_dist = self.create_field(container, "Distance (m):", str(record.get('distance_m', '')))
        self.entry_height = self.create_field(container, "Height (m):", str(record.get('height_m', '')))
        self.entry_time = self.create_field(container, "Timestamp:", record.get('timestamp', ''))
        self.entry_note = self.create_field(container, "Note", record.get('note', ''))

        self.lbl_status = ctk.CTkLabel(container, text="", text_color=COLOR_DANGER, font=FONT_BOLD)
        self.lbl_status.pack(pady=(0, 10))

        ctk.CTkButton(container, text="SAVE CHANGES", fg_color=COLOR_SUCCESS, height=40, font=FONT_BOLD, command=self.save_changes).pack(pady=30)

    def create_field(self, parent, label_text, value):
        ctk.CTkLabel(parent, text=label_text, anchor="w", font=FONT_BOLD).pack(fill="x", pady=(10,0))
        entry = ctk.CTkEntry(parent, width=350)
        entry.insert(0, str(value))
        entry.bind('<Return>', lambda event: self.save_changes())
        entry.pack(fill="x", pady=(5, 0))
        return entry
    
    def validate_decimal(self, value):
        """Checks if a string is a valid float with up to 2 decimal places."""
        try:
            # Check if it's a valid number
            float_val = float(value)
            # Use regex to ensure no more than 2 decimal places
            if re.match(r'^\d+(\.\d{1,2})?$', str(value)):
                return True
            return False
        except ValueError:
            return False
        
    def save_changes(self):
        new_plate = self.entry_plate.get().strip().upper().replace(" ", "")
        color = self.entry_color.get().strip()
        dist = self.entry_dist.get().strip()
        height = self.entry_height.get().strip()
        timestamp = self.entry_time.get().strip()
        note = self.entry_note.get().strip()

        # 2. Validation: Ensure required fields are not empty
        if not all([new_plate, color, dist, height, timestamp]):
            self.lbl_status.configure(text="Error: All fields except Note are required!")
            return

        # 3. Validation: Distance and Height maximal 2 decimal places
        if not self.validate_decimal(dist) or not self.validate_decimal(height):
            self.lbl_status.configure(text="Error: Distance/Height must be numbers (max 2 decimals)")
            return

        new_data = {
            'timestamp': timestamp,
            'plate_number': new_plate,
            'color': color,
            'distance_m': float(dist),
            'height_m': float(height),
            'note': note,
            'confidence': self.record.get('confidence', 1.0)
        }
        try:
            if new_plate != self.old_plate:
                ref.child('detection_logs').child(self.user_id).child(self.old_plate).delete()
                ref.child('detection_logs').child(self.user_id).child(new_plate).set(new_data)
            else:
                ref.child('detection_logs').child(self.user_id).child(self.old_plate).update(new_data)
            
            # print("✅ Record Updated")
            self.on_save() 
            self.destroy() 
        except Exception as e:
            pass # print(f"❌ Save Error: {e}")

# ==========================================
# PAGE: ADMIN DASHBOARD
# ==========================================
class AdminDashboard(ctk.CTkFrame):
    def __init__(self, master, on_logout):
        super().__init__(master)
        self.pack(fill="both", expand=True)

        # Header
        header = ctk.CTkFrame(self, height=80, fg_color=COLOR_CARD)
        header.pack(fill="x", padx=0, pady=0)
        
        ctk.CTkLabel(header, text="ADMIN DATABASE PANEL", font=FONT_HEADER).pack(side="left", padx=30, pady=20)

        btn_frame = ctk.CTkFrame(header, fg_color="transparent")
        btn_frame.pack(side="right", padx=20)
        
        ctk.CTkButton(btn_frame, text="REFRESH", width=120, command=self.load_all_data).pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text="LOGOUT", fg_color=COLOR_DANGER, width=100, command=on_logout).pack(side="left")

        # Content
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=20)

        # Table Header
        header_frame = ctk.CTkFrame(content, height=50, fg_color="#444", corner_radius=10)
        header_frame.pack(fill="x", pady=(0, 10), padx=(0, 20))
        
        cols = [("Username", 2), ("Email", 3), ("Registered", 2), ("Actions", 3)]
        for i, (col, w) in enumerate(cols):
            header_frame.grid_columnconfigure(i, weight=w, uniform="cols")
            ctk.CTkLabel(header_frame, text=col, font=FONT_BOLD, anchor="w").grid(row=0, column=i, sticky="ew", padx=10, pady=10)

        # Scrollable List
        self.scroll = ctk.CTkScrollableFrame(content, fg_color="transparent")
        self.scroll.pack(fill="both", expand=True)

        for i, (_, w) in enumerate(cols):
            self.scroll.grid_columnconfigure(i, weight=w)

        self.load_all_data()

    def load_all_data(self):
        for w in self.scroll.winfo_children(): w.destroy()
        users = db_manager.get_all_users()
        
        if not users:
            ctk.CTkLabel(self.scroll, text="No registered users found.", text_color="gray").pack(pady=20)
            return

        for idx, u in enumerate(users):
            self.create_row(idx, u)

    def create_row(self, row_idx, user_data):
        bg_color = COLOR_CARD if row_idx % 2 == 0 else "transparent"
        row_frame = ctk.CTkFrame(self.scroll, fg_color=bg_color, corner_radius=5)
        row_frame.grid(row=row_idx, column=0, columnspan=4, sticky="ew", pady=2)
        
        # Configure internal grid of the row frame to match parent
        weights = [2, 3, 2, 3]
        for i, w in enumerate(weights):
            row_frame.grid_columnconfigure(i, weight=w, uniform="cols")

        date_str = user_data['register_date'].split('.')[0]
        
        ctk.CTkLabel(row_frame, text=user_data['username'], anchor="w").grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        ctk.CTkLabel(row_frame, text=user_data['email'], anchor="w").grid(row=0, column=1, sticky="ew", padx=10, pady=10)
        ctk.CTkLabel(row_frame, text=date_str, anchor="w").grid(row=0, column=2, sticky="ew", padx=10, pady=10)
        
        action_frame = ctk.CTkFrame(row_frame, fg_color="transparent")
        action_frame.grid(row=0, column=3, sticky="ew", padx=10, pady=5)
        
        ctk.CTkButton(action_frame, text="DB", width=40, height=25, fg_color="#1E88E5", 
                      command=lambda uid=user_data['uid']: self.view_user_db(uid)).pack(side="left", padx=2)
        
        ctk.CTkButton(action_frame, text="Edit", width=50, height=25, fg_color=COLOR_WARNING, text_color="black",
                      command=lambda: self.open_edit_user(user_data)).pack(side="left", padx=2)

        ctk.CTkButton(action_frame, text="Del", width=50, height=25, fg_color=COLOR_DANGER,
                      command=lambda uid=user_data['uid']: self.confirm_delete(uid)).pack(side="left", padx=2)

    def view_user_db(self, user_id):
        UserHistoryWindow(self, user_id, enable_editing=True)

    def open_edit_user(self, user_data):
        EditUserWindow(self, user_data['uid'], user_data['username'], user_data['email'], self.load_all_data)

    def confirm_delete(self, user_id):
        confirm = ctk.CTkToplevel(self)
        confirm.title("Confirm Delete")
        confirm.geometry("300x180")
        confirm.attributes('-topmost', True)
        
        ctk.CTkLabel(confirm, text="Are you sure?\nThis will delete the user\nand ALL their logs.", 
                     font=FONT_BOLD, text_color=COLOR_DANGER).pack(pady=20)
        
        btn_frame = ctk.CTkFrame(confirm, fg_color="transparent")
        btn_frame.pack(fill="x", pady=10)
        
        ctk.CTkButton(btn_frame, text="Cancel", width=100, fg_color="gray", command=confirm.destroy).pack(side="left", padx=20)
        ctk.CTkButton(btn_frame, text="DELETE", width=100, fg_color=COLOR_DANGER, 
                      command=lambda: self.perform_delete(user_id, confirm)).pack(side="right", padx=20)

    def perform_delete(self, user_id, popup):
        db_manager.delete_full_user_data(user_id)
        popup.destroy()
        self.load_all_data()

# ==========================================
# PAGE: USER DASHBOARD (HISTORY)
# ==========================================
class UserHistoryWindow(ctk.CTkToplevel):
    def __init__(self, master, user_id, enable_editing=False, filter_source=None):
        super().__init__(master)
        self.attributes('-topmost', True)
        self.geometry("1400x700")

        self.filter_source = filter_source
        self.user_id = user_id
        self.enable_editing = enable_editing

        display_title = f"Logs: {filter_source}" if filter_source else "All Logs"
        self.title(f"{display_title} - {user_id}")

        # Top Bar
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=20, pady=20)
        
        ctk.CTkLabel(top, text=display_title, font=FONT_HEADER).pack(side="left")
        ctk.CTkButton(top, text="Refresh Data", command=self.load_data, fg_color=COLOR_ACCENT).pack(side="right")

        # Header Row
        headers = ["Plate", "Time", "Source", "Color", "Confidence", "Dist/Height", "Note"]
        weights = [1, 2, 2, 1, 1, 2, 3]
        if self.enable_editing:
            headers.append("Action")
            weights.append(1)

        header_frame = ctk.CTkFrame(self, fg_color=COLOR_CARD, height=40)
        header_frame.pack(fill="x", padx=20)

        for i, h in enumerate(headers):
            header_frame.grid_columnconfigure(i, weight=weights[i])
            ctk.CTkLabel(header_frame, text=h, font=FONT_BOLD).grid(row=0, column=i, sticky="ew", pady=10)

        # List
        self.scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll.pack(fill="both", expand=True, padx=20, pady=10)
        
        for i, w in enumerate(weights):
            self.scroll.grid_columnconfigure(i, weight=w)

        self.load_data()

    def load_data(self):
        for w in self.scroll.winfo_children(): w.destroy()

        if not ref: return

        logs = ref.child('detection_logs').child(self.user_id).get()
        if not logs: 
            ctk.CTkLabel(self.scroll, text="No records found.").grid(row=0, column=0, columnspan=8, pady=20)
            return

        sorted_rows = sorted(logs.items(), key=lambda x: x[1].get('timestamp',''), reverse=True)

        idx = 0
        for plate, data in sorted_rows:
            if self.filter_source:
                log_source = data.get('camera_source', 'Unknown')
                if log_source != self.filter_source:
                    continue
            self.create_row(idx, plate, data)
            idx += 1

    def create_row(self, row, plate, record):
        # Map logical row index to grid rows (2 rows per record: content + separator)
        grid_row = row * 2
        sep_row = grid_row + 1

        bg_color = "#333" if row % 2 == 0 else "transparent"
        
        # We need a frame for the background color effect, but grid makes it tricky with columns.
        # So we just add labels directly but perhaps we can put them in a frame wrapper later.
        # For simple list, let's just stick to direct grid on scroll frame with separators.

        def cell(c, txt, color="white"):
            lbl = ctk.CTkLabel(self.scroll, text=str(txt), text_color=color, anchor="center")
            lbl.grid(row=grid_row, column=c, sticky="ew", pady=5)
            
        cell(0, plate, COLOR_WARNING)
        cell(1, record.get('timestamp', '-'))
        cell(2, record.get('camera_source', '-'))
        cell(3, record.get('color', '-'))
        cell(4, f"{record.get('confidence', 0):.2f}")
        cell(5, f"{record.get('distance_m', 0)}m / {record.get('height_m', 0)}m")
        cell(6, record.get('note', ''))

        if self.enable_editing:
            btn = ctk.CTkButton(self.scroll, text="EDIT", width=60, height=25, fg_color=COLOR_WARNING, text_color="black",
                                command=lambda p=plate, d=record: self.open_edit(p, d))
            btn.grid(row=grid_row, column=7, pady=5)
        
        # Add a separator line
        sep = ctk.CTkFrame(self.scroll, height=1, fg_color="#444")
        sep.grid(row=sep_row, column=0, columnspan=8, sticky="ew")

    def open_edit(self, plate, record):
        EditRecordWindow(self, self.user_id, plate, record, self.load_data)

# ==========================================
# POPUP: EDIT USER PROFILE
# ==========================================
class EditUserWindow(ctk.CTkToplevel):
    def __init__(self, master, target_user_id, current_username, current_email, on_save_callback):
        super().__init__(master)
        self.attributes('-topmost', True)
        self.grab_set() 
        self.title(f"Edit Profile")
        self.geometry("400x450")
        
        self.target_id = target_user_id
        self.on_save = on_save_callback

        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(container, text="Update Information", font=FONT_SUBHEADER).pack(pady=(0, 20))

        ctk.CTkLabel(container, text="Username:", font=FONT_BOLD, anchor="w").pack(fill="x", pady=(5,0))
        self.entry_user = ctk.CTkEntry(container)
        self.entry_user.insert(0, current_username)
        self.entry_user.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(container, text="Email:", font=FONT_BOLD, anchor="w").pack(fill="x", pady=(5,0))
        self.entry_email = ctk.CTkEntry(container)
        self.entry_email.insert(0, current_email)
        self.entry_email.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(container, text="New Password:", font=FONT_BOLD, anchor="w").pack(fill="x", pady=(5,0))
        self.entry_pass = ctk.CTkEntry(container, placeholder_text="Leave empty to keep current", show="*")
        self.entry_pass.pack(fill="x", pady=(0, 10))

        self.lbl_status = ctk.CTkLabel(container, text="", text_color=COLOR_WARNING)
        self.lbl_status.pack(pady=5)

        ctk.CTkButton(container, text="SAVE CHANGES", fg_color=COLOR_SUCCESS, height=40, font=FONT_BOLD, 
                      command=self.attempt_save).pack(pady=20)

    def validate_email(self, email):
        # Strict Regex for Email Format
        pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        return re.match(pattern, email) is not None

    def attempt_save(self):
        u = self.entry_user.get().strip()
        e = self.entry_email.get().strip()
        p = self.entry_pass.get().strip()

        if not u:
            self.lbl_status.configure(text="Username cannot be empty", text_color=COLOR_DANGER)
            return
        
        # 2. Email Empty Check
        if not e:
            self.lbl_status.configure(text="Email cannot be empty", text_color=COLOR_DANGER)
            return

        # 3. Email Format Validation
        if not self.validate_email(e):
            self.lbl_status.configure(text="Invalid email format!", text_color=COLOR_DANGER)
            return
        
        success, msg = db_manager.update_user_info(self.target_id, u, e, p)
        
        if success:
            self.on_save() 
            self.destroy()
        else:
            self.lbl_status.configure(text=msg, text_color=COLOR_DANGER)

# ==========================================
# PAGE: MAIN LPR DASHBOARD
# ==========================================
class DashboardFrame(ctk.CTkFrame):
    def __init__(self, master, camera_source, user_id, on_close): 
        super().__init__(master)
        self.pack(fill="both", expand=True)
        self.on_close_callback = on_close 
        self.camera_ip = camera_source
        self.user_id = user_id 
        self.is_running = True
        self.cap = None 

        # print("Loading AI Models...")
        self.detector = YOLO(resource_path("best.pt"))
        self.color_model = YOLO(resource_path("color.pt"))
        # ADDED verbose=False to silence EasyOCR
        self.reader = easyocr.Reader(['en'], gpu=True, verbose=False)
        self.ALLOW_LIST = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                
        self.init_logic_variables()
        self.create_layout()
        
        self.download_path = os.path.join(os.path.expanduser("~"), "Downloads", "SmartLPR_Backup")
        self.img_folder = os.path.join(self.download_path, "captured_images")
        os.makedirs(self.img_folder, exist_ok=True)
        self.csv_filename = os.path.join(self.download_path, 'car_plate_records.csv')
        # print(f"📂 Backup Folder: {self.download_path}")

        threading.Thread(target=self.connect_camera, daemon=True).start()
        
        self.csv_headers = [
            "Timestamp", 
            "Plate Number", 
            "Confidence", 
            "Color", 
            "Distance (m)", 
            "Height (m)", 
            "Note"
        ]
        
        if not os.path.isfile(self.csv_filename):
            try:
                with open(self.csv_filename, 'w', newline='') as f:
                    csv.writer(f).writerow(self.csv_headers)
            except Exception as e:
                """print(f"Error creating CSV headers: {e}")"""


            
    def connect_camera(self):
        try:
            # 1. Attempt connection (Blocking happens here, but it's safe now)
            if self.camera_ip.isdigit():
                src = int(self.camera_ip)
                cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(self.camera_ip)

            # 2. Check success
            if not cap.isOpened():
                raise ValueError("Could not open video source")
            
            # 3. If successful, assign to self.cap and start loop
            self.cap = cap
            # Schedule the update loop on the main thread
            self.after(0, self.update_camera)
            
        except Exception as e:
            # print(f"Camera Init Error: {e}")
            # Update UI label safely from thread
            self.after(0, lambda: self.video_label.configure(text=f"Connection Failed:\n{e}", text_color=COLOR_DANGER))

    def create_layout(self):
            self.grid_columnconfigure(1, weight=1)
            self.grid_rowconfigure(0, weight=1)

            # Sidebar
            self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0, fg_color="#222")
            self.sidebar.grid(row=0, column=0, sticky="nsew")
            
            ctk.CTkLabel(self.sidebar, text="LIVE MONITORING", font=FONT_HEADER, text_color="white").pack(pady=(30, 20))
            
            self.lbl_plate = self.create_card("Detected Plate", "---", COLOR_WARNING)
            self.lbl_color = self.create_card("Vehicle Color", "---", "white")
            self.lbl_dist = self.create_card("Dist / Height", "- / -", "white")
            
            # Controls
            ctrl_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
            ctrl_frame.pack(fill="x", side="bottom", pady=30, padx=20)

            ctk.CTkButton(ctrl_frame, text="⚙ SETTINGS", fg_color="#444", height=40, font=FONT_BOLD, command=lambda: SettingsWindow(self)).pack(fill="x", pady=5)
            # NEW (Correct - filters by the current camera IP)
            ctk.CTkButton(ctrl_frame, text="📂 HISTORY", fg_color=COLOR_ACCENT, height=40, font=FONT_BOLD, command=lambda: UserHistoryWindow(self, self.user_id, filter_source=self.camera_ip)).pack(fill="x", pady=5)
            self.btn_manual = ctk.CTkButton(ctrl_frame, text="✎ MANUAL INPUT", fg_color=COLOR_WARNING, text_color="black", height=40, font=FONT_BOLD, command=self.manual_correction_popup)
            self.btn_manual.pack(fill="x", pady=5)
            
            ctk.CTkButton(ctrl_frame, text="⏹ STOP / BACK", fg_color=COLOR_DANGER, height=40, font=FONT_BOLD, command=self.on_close_callback).pack(fill="x", pady=(20, 0))

            # Video Area
            self.video_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="black")
            self.video_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
            
            # Make the label fill the frame
            self.video_label = ctk.CTkLabel(self.video_frame, text="Loading Camera Feed...", text_color="gray", font=FONT_HEADER)
            self.video_label.pack(expand=True, fill="both")

    def create_card(self, title, default, color):
        f = ctk.CTkFrame(self.sidebar, fg_color="#333", corner_radius=10)
        f.pack(padx=20, pady=10, fill="x")
        ctk.CTkLabel(f, text=title.upper(), font=("Roboto", 11, "bold"), text_color="gray").pack(anchor="w", padx=15, pady=(10, 0))
        l = ctk.CTkLabel(f, text=default, font=("Roboto", 22, "bold"), text_color=color)
        l.pack(anchor="w", padx=15, pady=(0, 10))
        return l

    def init_logic_variables(self):
        self.plate_buffer = []; self.conf_buffer = []; self.color_buffer = []
        self.dist_buffer = []; self.height_buffer = []
        self.BUFFER_SIZE = 5; self.last_saved_time = datetime.datetime.min
        self.COOLDOWN_SECONDS = 15; self.frame_count = 0
        self.current_detections = []
        self.last_known_color = "Unknown"; self.last_known_dist = 0.0; self.last_known_height = 0.0
        self.last_saved_plate_key = None
        self.current_clean_frame = None

    def process_logic(self, frame):
        h_img, w_img, _ = frame.shape
        line_y = SystemConfig.get_trigger_y(h_img)

        self.current_clean_frame=frame.copy()
        overlay = frame.copy()
        cv2.line(frame, (0, line_y), (w_img, line_y), (0, 100, 255), 2)
        alpha = SystemConfig.LINE_OPACITY
        cv2.addWeighted(frame, alpha, overlay, 1 - alpha, 0, frame)

        if self.frame_count % 5 == 0:
            results = self.detector.predict(frame, conf=SystemConfig.CONFIDENCE_THRESHOLD, verbose=False)
            self.current_detections = []
            current_ocr = None
            current_conf = 0.0

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    
                    if cls_id == 0: # Car
                        w_box, h_box = x2-x1, y2-y1
                        dist, real_h = estimate_distance_and_size(w_box, h_box)
                        self.last_known_dist = dist
                        self.last_known_height = real_h
                        
                        if w_box > 50:
                            try:
                                car_crop = frame[y1:y2, x1:x2]
                                color_res = self.color_model.predict(car_crop, conf=SystemConfig.CONFIDENCE_THRESHOLD, verbose=False)
                                self.last_known_color = color_res[0].names[color_res[0].probs.top1]
                            except: pass
                        
                        self.current_detections.append([x1,y1,x2,y2, 0, f"{self.last_known_color}", dist])

                    elif cls_id == 1: # Plate
                        cy = (y1 + y2) // 2
                        if (line_y - 100) < cy < (line_y + 100):
                            plate_crop = frame[y1:y2, x1:x2]
                            clean = preprocess_plate(plate_crop)
                            ocr_res = self.reader.readtext(clean, allowlist=self.ALLOW_LIST)
                            if ocr_res:
                                detections = sorted(
                                    [res for res in ocr_res if res[2] > 0.6 and len(res[1].strip()) > 1],
                                    key=lambda res: res[0][0][0]
                                )
                                if detections:
                                    texts = [res[1].upper() for res in detections]
                                    txt = "".join(texts)
                                    conf = max(res[2] for res in detections)
                                    if conf > 0.4:
                                        current_ocr = auto_correct_plate(txt)
                                        current_conf = conf

                                        is_vanity = False
                                        raw_upper = txt.upper()
                                        for vp in VANITY_PREFIXES:
                                            if raw_upper.startswith(vp) or vp in raw_upper[:len(vp) + 4]:
                                                is_vanity = True
                                                break

                                        if is_vanity:
                                            current_ocr = txt.replace('0', 'O').replace('1', 'I')
                                            # print(f"Vanity plate detected: {current_ocr} (raw trusted)")
                                        else:
                                            current_ocr = auto_correct_plate(txt)
                                            # print(f"Normal plate corrected: {current_ocr}")

                                        self.current_detections.append([x1,y1,x2,y2, 1, current_ocr, 0])
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 3)

            # SAVE LOGIC
            if current_ocr:
                self.plate_buffer.append(current_ocr)
                self.conf_buffer.append(current_conf)
                self.color_buffer.append(self.last_known_color)
                self.dist_buffer.append(self.last_known_dist)
                self.height_buffer.append(self.last_known_height)
                
                if len(self.plate_buffer) > self.BUFFER_SIZE:
                    self.plate_buffer.pop(0); self.conf_buffer.pop(0)
                    self.color_buffer.pop(0); self.dist_buffer.pop(0); self.height_buffer.pop(0)

                if len(self.plate_buffer) == self.BUFFER_SIZE:
                    top_plate, count = Counter(self.plate_buffer).most_common(1)[0]
                    if count >= 3:
                        now = datetime.datetime.now()
                        if (now - self.last_saved_time).total_seconds() > self.COOLDOWN_SECONDS:
                            self.save_record(now, top_plate, self.conf_buffer[0], self.color_buffer[0], self.dist_buffer[0], self.height_buffer[0], image=self.current_clean_frame)
                            # print(self.plate_buffer)
                            self.last_saved_time = now
                            self.plate_buffer = []

        # Draw
        for x1, y1, x2, y2, cid, lbl, d in self.current_detections:
            color = (255,255,0) if cid==0 else (0,255,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, lbl, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def manual_correction_popup(self):
        dialog = ctk.CTkInputDialog(text="Enter Correct Plate Number:", title="Manual Correction")
        manual_plate = dialog.get_input()

        if manual_plate:
            new_plate = manual_plate.upper().replace(" ", "")
            now = datetime.datetime.now()
            
            current_color = self.last_known_color if self.last_known_color != "Unknown" else "Manual_Color"
            current_dist = self.last_known_dist
            current_height = self.last_known_height

            self.lbl_plate.configure(text=f"{new_plate} (M)")
            
            if ref and self.user_id:
                try:
                    if self.last_saved_plate_key:
                        ref.child('detection_logs').child(self.user_id).child(self.last_saved_plate_key).delete()
                        
                        data = {
                            'timestamp': now.strftime("%Y-%m-%d %H:%M:%S"),
                            'camera_source': self.camera_ip,
                            'plate_number': new_plate,
                            'confidence': 1.0,
                            'color': self.last_known_color,
                            'distance_m': f"{self.last_known_dist:.2f}",
                            'height_m': f"{self.last_known_height:.2f}",
                            'note': "Manually Corrected"
                        }
                        ref.child('detection_logs').child(self.user_id).child(new_plate).set(data)
                        self.last_saved_plate_key = new_plate
                        # print(f"✅ Manual Update: {new_plate}")
                except Exception as e:
                    pass # print(f"Update failed: {e}")

            with open(self.csv_filename, 'a', newline='') as f:
                csv.writer(f).writerow([
                    now.strftime("%Y-%m-%d %H:%M:%S"), 
                    new_plate, 
                    "MANUAL_CORRECTION", 
                    current_color, 
                    f"{current_dist:.2f}", 
                    f"{current_height:.2f}", 
                    "Previous_Record_Overridden"
                ])

    def save_record(self, time_obj, plate, conf, color, dist, height, note="", image=None):
        self.lbl_plate.configure(text=plate)
        self.lbl_color.configure(text=color)
        self.lbl_dist.configure(text=f"{dist:.1f}m / {height:.1f}m")

        with open(self.csv_filename, 'a', newline='') as f:
            csv.writer(f).writerow([time_obj, plate, f"{conf:.2f}", color, f"{dist:.2f}", f"{height:.2f}", note])   

        if image is not None:
            try:
                timestamp_str = time_obj.strftime("%Y%m%d_%H%M%S")
                img_name = f"{plate}_{timestamp_str}.jpg"
                save_path = os.path.join(self.img_folder, img_name)
                cv2.imwrite(save_path, image)
                # print(f"📸 Image Saved: {save_path}")
            except Exception as e:
                pass # print(f"⚠️ Image Save Failed: {e}")

        if ref and self.user_id:
            data = {
                'timestamp': time_obj.strftime("%Y-%m-%d %H:%M:%S"),
                'camera_source': self.camera_ip,
                'plate_number': plate,
                'confidence': float(f"{conf:.2f}"),
                'color': color,
                'distance_m': float(f"{dist:.2f}"),
                'height_m': float(f"{height:.2f}")
            }
            try:
                ref.child('detection_logs').child(self.user_id).child(plate).set(data)
                self.last_saved_plate_key = plate
                # print(f"Uploaded: {plate} for User {self.camera_ip}")
            except Exception as e:
                pass # print(f"Cloud Error: {e}")

    def update_camera(self):
        if not self.is_running: return
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            frame = cv2.resize(frame, (640, 640))
            frame = self.process_logic(frame)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            w = self.video_frame.winfo_width()
            h = self.video_frame.winfo_height()
            
            # Keep aspect ratio? For now fill
            if w > 10 and h > 10: 
                img = img.resize((w, h), Image.Resampling.LANCZOS)
                
            imgtk = ctk.CTkImage(img, size=(w, h))
            self.video_label.configure(image=imgtk, text="")
            self.video_label.image = imgtk
        self.after(10, self.update_camera)

    def stop_and_exit(self):
        self.is_running = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

# ==========================================
# APP CONTROLLER
# ==========================================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Smart LPR System")
        self.geometry("1200x800")
        
        # Main Container
        self.container = ctk.CTkFrame(self, fg_color="#1a1a1a")
        self.container.pack(fill="both", expand=True)
        
        self.current_frame = None
        self.last_user_data = None
        self.show_login()

    def show_register(self):
        self.clear_frame()
        self.current_frame = RegisterFrame(self.container, self.show_login)

    def on_login_success(self, role, user_data):
        if role == "ADMIN":
            self.start_admin()
        else:
            self.show_camera_selection(user_data)

    def show_login(self):
        self.clear_frame()
        self.current_frame = LoginFrame(self.container, self.on_login_success, self.show_register)

    def show_camera_selection(self, user_data):
        self.last_user_data = user_data
        self.clear_frame()
        self.current_frame = CameraSelectionFrame(self.container, user_data, self.start_dashboard, self.show_login)

    def start_dashboard(self, ip):
        self.clear_frame()
        uid = self.last_user_data[0]
        self.current_frame = DashboardFrame(self.container, ip, uid, lambda: self.show_camera_selection(self.last_user_data))

    def start_admin(self):
        self.clear_frame()
        self.current_frame = AdminDashboard(self.container, self.show_login)

    def clear_frame(self):
        if self.current_frame:
            if hasattr(self.current_frame, 'stop_and_exit'):
                self.current_frame.stop_and_exit()
            self.current_frame.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()