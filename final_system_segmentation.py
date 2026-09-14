import firebase_admin
from firebase_admin import credentials, db
import hashlib
import datetime
from LRP_system import resource_path
# --- CONFIGURATION ---
CRED_PATH = "serviceAccountKey.json"
# Your specific Database URL
DB_URL = ''

class FirebaseManager:
    def __init__(self):
        # 1. INITIALIZE FIREBASE
        if not firebase_admin._apps:
            cred = credentials.Certificate(resource_path(CRED_PATH))
            firebase_admin.initialize_app(cred, {
                'databaseURL': DB_URL
            })
        
        self.ref = db.reference()
        
        self.ensure_admin_exists()

    def hash_password(self, password):
        # Secure password hashing
        return hashlib.sha256(password.encode()).hexdigest()

    def ensure_admin_exists(self):
        # Check if 'admins' node exists
        admins = self.ref.child('admins').get()
        if not admins:
            print("Creating default admin account (admin/admin123)...")
            self.register_admin("admin", "admin123")

    # --- USER FUNCTIONS ---
    def register_user(self, username, password, email):
        users_ref = self.ref.child('users')
    
        snapshot = users_ref.order_by_child('username').equal_to(username).get()
        if snapshot:
            return False, "Username already exists"

        new_user_ref = users_ref.child(username)
        new_user_ref.set({
            'username': username,
            'password': self.hash_password(password),
            'email': email,
            'register_date': str(datetime.datetime.now())
        })
        return True, "Registration Successful"

        
    def login_user(self, username, password):
        hashed_pw = self.hash_password(password)
        users = self.ref.child('users').order_by_child('username').equal_to(username).get()
        
        if users:
            for uid, data in users.items():
                if data['password'] == hashed_pw:
                    return (uid, data['username'])
        return None

    # --- ADMIN FUNCTIONS ---
    def register_admin(self, username, password):
        self.ref.child('admins').push().set({
            'username': username,
            'password': self.hash_password(password),
            'created': str(datetime.datetime.now())
        })

    def login_admin(self, username, password):
        hashed_pw = self.hash_password(password)
        admins = self.ref.child('admins').get()
        if admins:
            for aid, data in admins.items():
                if data.get('username') == username and data.get('password') == hashed_pw:
                    return (aid, username)
        return None

    # --- CAMERA FUNCTIONS ---
    def add_camera(self, user_id, ip_address):
        # Store camera under /cameras/{user_id}/{camera_id}
        new_cam_ref = self.ref.child('cameras').child(user_id).push()
        new_cam_ref.set({
            'ip_address': ip_address,
            'created': str(datetime.datetime.now())
        })

    def get_user_cameras(self, user_id):
        cameras = self.ref.child('cameras').child(user_id).get()
        cam_list = []
        if cameras:
            for cid, data in cameras.items():
                cam_list.append((cid, data.get('ip_address', 'Unknown')))
        return cam_list

    def delete_camera(self, user_id, camera_id):
        self.ref.child('cameras').child(user_id).child(camera_id).delete()

    def get_all_users(self):
        """Fetch all registered users for the Admin List."""
        users = self.ref.child('users').get()
        user_list = []
        if users:
            for uid, data in users.items():
                user_list.append({
                    'uid': uid,
                    'username': data.get('username', 'Unknown'),
                    'email': data.get('email', '-'),
                    'register_date': data.get('register_date', '-')
                })
        return user_list

    def delete_full_user_data(self, user_id):
        """
        Delete EVERYTHING related to this user:
        1. The User Account
        2. Their Cameras
        3. Their Detection Logs
        """
        try:
            self.ref.child('users').child(user_id).delete()
            self.ref.child('cameras').child(user_id).delete()
            self.ref.child('detection_logs').child(user_id).delete()
            return True
        except Exception as e:
            print(f"Error deleting user: {e}")
            return False


    def update_user_info(self, user_id, new_username, new_email, new_password=None):
        users_ref = self.ref.child('users')
        
        # 1. Get current data to compare
        current_data = users_ref.child(user_id).get()
        if not current_data:
            return False

        # 2. Check Username Uniqueness (Only if changed)
        if new_username != current_data.get('username'):
            snapshot = users_ref.order_by_child('username').equal_to(new_username).get()
            if snapshot:
                return False

        # 3. Prepare Update Data
        update_packet = {
            'username': new_username,
            'email': new_email
        }
        
        # Only update password if user typed something new
        if new_password and len(new_password) > 0:
            update_packet['password'] = self.hash_password(new_password)

        # 4. Perform Update
        try:
            users_ref.child(user_id).update(update_packet)
            return True, "Profile Updated Successfully"
        except Exception as e:
            return False, str(e)
        
# --- INSTANTIATE IMMEDIATELY ---
try:
    db_manager = FirebaseManager()
    ref = db_manager.ref
    #print("✅ Database Connected via final_system_segmentation.py")
except Exception as e:
    print(f"❌ Database Connection Error: {e}")
    db_manager = None
    ref = None