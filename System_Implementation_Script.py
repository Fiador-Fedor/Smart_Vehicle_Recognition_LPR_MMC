import os
import io
import time
import datetime
import uuid
import threading
import traceback

# Computer vision / ML imports (from your final_integrate.py)
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Raspberry Pi hardware
try:
    import RPi.GPIO as GPIO
    from picamera import PiCamera
    from picamera.array import PiRGBArray
    ON_PI = True
except Exception:
    ON_PI = False
    # If not running on Pi, fallback to cv2.VideoCapture
    pass

# Firebase admin
import firebase_admin
from firebase_admin import credentials, storage, firestore

# ---------------- Config ----------------
SERVICE_ACCOUNT_PATH = "/home/pi/firebase-service-account.json"  # <-- put your service account here
FIREBASE_STORAGE_BUCKET = "your-project-id.appspot.com"         # <-- change to your bucket
FIREBASE_PROJECT_ID = "your-project-id"                        # <-- change to your project id

CAPTURE_INTERVAL = 5  # seconds between automated captures (can be lowered)
UPLOAD_OUTPUT_DIR = "output"  # local copies

# GPIO pins (BOARD numbering in earlier code)
RED_GROUP = 13
GREEN_GROUP = 15

# Optional sensor pin (PIR example)
PIR_PIN = 11  # physical pin 11 (GPIO17)

# ---------------- YOLO + OCR SETUP (from your final_integrate.py) ----------------
plate_detector = YOLO("yolov11n_detect.pt")
ocr_model = load_model("final_model.keras")

# color classifier setup (same as your file)
colour_labels = ['beige', 'black', 'blue', 'brown', 'gold', 'green', 'grey',
                 'orange', 'pink', 'purple', 'red', 'silver', 'tan', 'white', 'yellow']
num_classes = len(colour_labels)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

color_model = models.resnet50(pretrained=False)
color_model.fc = nn.Linear(color_model.fc.in_features, num_classes)
color_model = color_model.to(device)
color_model.load_state_dict(torch.load("vcor_finetuned_resnet50_final.pth", map_location=device))
color_model.eval()

color_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# make & model
make_model = load_model("stanford_car_resnet50_model.keras")
with open("car_classes.txt") as f:
    make_model_classes = [line.strip() for line in f]
IMG_SIZE = (224, 224)

# --- OCR helpers (same as your file) ---
def find_contours(dimensions, img):
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lower_width, upper_width, lower_height, upper_height = dimensions
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    x_cntr_list, img_res = [], []
    for cntr in cntrs:
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        if lower_width < intWidth < upper_width and lower_height < intHeight < upper_height:
            char_copy = np.zeros((44, 24))
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            char = cv2.subtract(255, char)
            char_copy[2:42, 2:22] = char
            img_res.append(char_copy)
            x_cntr_list.append(intX)
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res = np.array([img_res[i] for i in indices]) if len(indices) > 0 else np.array([])
    return img_res


def segment_characters(image):
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # kernels should be provided as numpy arrays; original code used tuples — keep original intent
    kernel = np.ones((3,3), np.uint8)
    img_binary_lp = cv2.erode(img_binary_lp, kernel)
    img_binary_lp = cv2.dilate(img_binary_lp, kernel)
    LP_WIDTH, LP_HEIGHT = img_binary_lp.shape[0], img_binary_lp.shape[1]
    img_binary_lp[0:3, :] = 255; img_binary_lp[:, 0:3] = 255
    img_binary_lp[72:75, :] = 255; img_binary_lp[:, 330:333] = 255
    dimensions = [LP_WIDTH/6, LP_WIDTH/2, LP_HEIGHT/10, 2*LP_HEIGHT/3]
    return find_contours(dimensions, img_binary_lp)

def fix_dimension(img):
    new_img = np.zeros((28,28,3))
    for i in range(3): new_img[:,:,i] = img
    return new_img

def ocr_predict(char_list):
    dic = {i: c for i, c in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
    output = []
    if char_list is None or len(char_list) == 0:
        return ""
    for ch in char_list:
        img_ = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3)
        y_ = np.argmax(ocr_model.predict(img), axis=-1)[0]
        output.append(dic[y_])
    return ''.join(output)

def predict_color(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    input_tensor = color_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = color_model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        top_prob, top_class = torch.max(probs, 1)
    return colour_labels[top_class.item()], float(top_prob.item() * 100)

def predict_make_model(image_bgr):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(pil_img), axis=0)
    img_array = img_array.astype("float32")
    preds = make_model.predict(img_array)
    idx = np.argmax(preds)
    return make_model_classes[idx], float(preds[0][idx] * 100)

def detect_and_read_plate_from_image(image):
    """
    Accepts an OpenCV BGR image (numpy array).
    Returns: dict with plate, color, make_model, confidences, annotated_image (BGR)
    """
    results = plate_detector(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        return None

    x1, y1, x2, y2 = boxes[0].astype(int)
    plate_crop = image[y1:y2, x1:x2]

    try:
        char_images = segment_characters(plate_crop)
        plate_number = ocr_predict(char_images) if len(char_images) > 0 else "UNKNOWN"
    except Exception as e:
        print("OCR error:", e)
        plate_number = "UNKNOWN"

    try:
        predicted_colour, colour_conf = predict_color(image)
    except Exception as e:
        print("Color error:", e)
        predicted_colour, colour_conf = "UNKNOWN", 0.0

    try:
        make_model_pred, mm_conf = predict_make_model(image)
    except Exception as e:
        print("Make/Model error:", e)
        make_model_pred, mm_conf = "UNKNOWN", 0.0

    annotated = results[0].plot()
    cv2.putText(annotated, f"{plate_number} | {predicted_colour} ({colour_conf:.1f}%) | {make_model_pred} ({mm_conf:.1f}%)",
                (max(5, x1), max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return {
        "plate": plate_number,
        "color": predicted_colour,
        "color_conf": float(colour_conf),
        "make_model": make_model_pred,
        "mm_conf": float(mm_conf),
        "annotated_image": annotated,
        "bbox": [int(x1), int(y1), int(x2), int(y2)]
    }

# ---------------- Firebase init ----------------
def init_firebase():
    if not os.path.exists(SERVICE_ACCOUNT_PATH):
        raise FileNotFoundError(f"Service account JSON not found: {SERVICE_ACCOUNT_PATH}")

    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred, {
        'storageBucket': FIREBASE_STORAGE_BUCKET,
        'projectId': FIREBASE_PROJECT_ID
    })
    db = firestore.client()
    bucket = storage.bucket()
    return db, bucket

# ---------------- Verification helpers ----------------
def verify_plate_with_firebase(db, scan_doc):
    """
    Checks the vehicles collection to determine if this scan should be treated as VERIFIED.
    - First checks if the vehicle doc exists and has 'verified': True.
    - Otherwise tries a fallback match: exact (case-insensitive) color & make/model or high-confidence single-field match.
    Returns boolean.
    """
    plate = scan_doc.get("licensePlate", "UNKNOWN")
    if not plate or plate.upper() == "UNKNOWN":
        return False

    plate_id = plate.replace(" ", "_").upper()
    vehicles_ref = db.collection("vehicles")
    doc_ref = vehicles_ref.document(plate_id)
    snap = doc_ref.get()
    if not snap.exists:
        return False

    data = snap.to_dict() or {}

    # 1) If firebase doc explicitly marks verified flag, respect it.
    if data.get("verified", None) is True:
        return True

    # 2) Fallback matching: compare color and make_model case-insensitively
    stored_color = (data.get("color") or "").strip().lower()
    stored_mm = (data.get("make_model") or "").strip().lower()
    scan_color = (scan_doc.get("color") or "").strip().lower()
    scan_mm = (scan_doc.get("make_model") or "").strip().lower()

    color_match = (stored_color != "" and stored_color == scan_color)
    mm_match = (stored_mm != "" and stored_mm == scan_mm)

    # If both match -> verified
    if color_match and mm_match:
        return True

    # Allow single-field high confidence matches:
    if color_match and float(scan_doc.get("colorConf", 0.0)) >= 60.0:
        return True
    if mm_match and float(scan_doc.get("mm_conf", 0.0)) >= 60.0:
        return True

    # Otherwise, flagged (not verified)
    return False

# ---------------- Upload helpers ----------------
def upload_image_to_storage(bucket, image_bgr, dest_path):
    # Convert BGR to JPEG bytes
    is_success, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not is_success:
        raise RuntimeError("Failed to encode image for upload")
    img_bytes = buffer.tobytes()

    blob = bucket.blob(dest_path)
    blob.upload_from_string(img_bytes, content_type='image/jpeg')
    # Make public or return signed URL as needed; here we return a public URL (if default ACL allows)
    try:
        blob.make_public()
        return blob.public_url
    except Exception:
        # fallback: generate signed URL valid for 7 days
        url = blob.generate_signed_url(expiration=datetime.timedelta(days=7))
        return url

def write_scan_to_firestore(db, scan_doc):
    scans_ref = db.collection("scans")
    doc_ref = scans_ref.document(scan_doc["id"])
    doc_ref.set(scan_doc)
    return doc_ref.id

def upsert_vehicle_from_scan(db, scan_doc, verified=False):
    """
    Update or create the vehicle document using the scan. Include verified flag if passed.
    """
    vehicles_ref = db.collection("vehicles")
    plate = (scan_doc.get("licensePlate") or "UNKNOWN")
    plate_id = plate.replace(" ", "_").upper()
    vehicle_doc_ref = vehicles_ref.document(plate_id)

    update_payload = {
        "licensePlate": plate,
        "make_model": scan_doc.get("make_model"),
        "color": scan_doc.get("color"),
        "lastScanAt": firestore.SERVER_TIMESTAMP,
        "scanCount": firestore.Increment(1),
        # store verification status as well
        "verified": verified
    }
    vehicle_doc_ref.set(update_payload, merge=True)
    return plate_id

# ---------------- GPIO & Pi camera ----------------
def setup_gpio():
    if not ON_PI:
        print("Not running on Pi: GPIO disabled.")
        return

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(RED_GROUP, GPIO.OUT)
    GPIO.setup(GREEN_GROUP, GPIO.OUT)
    try:
        GPIO.setup(PIR_PIN, GPIO.IN)
    except Exception:
        pass
    # Default off
    GPIO.output(RED_GROUP, False)
    GPIO.output(GREEN_GROUP, False)
    print("GPIO initialized (LEDs & PIR)")

def set_led(green_on=False, red_on=False):
    if not ON_PI:
        print(f"[LED SIM] green={green_on} red={red_on}")
        return
    GPIO.output(GREEN_GROUP, green_on)
    GPIO.output(RED_GROUP, red_on)

# ---------------- Capture loop ----------------
def capture_loop(db, bucket):
    # camera setup
    if ON_PI:
        camera = PiCamera()
        camera.resolution = (1280, 720)
        raw_capture = PiRGBArray(camera, size=camera.resolution)
        time.sleep(2)  # warmup
        stream = camera.capture_continuous(raw_capture, format="bgr", use_video_port=True)
        stream_iter = iter(stream)
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera")

    last_capture = 0
    try:
        while True:
            # Sensor-triggering (PIR)
            sensor_triggered = False
            if ON_PI:
                try:
                    if GPIO.input(PIR_PIN) == GPIO.HIGH:
                        sensor_triggered = True
                except Exception:
                    # PIR might not be wired — ignore
                    pass

            now = time.time()
            if (now - last_capture) < CAPTURE_INTERVAL and not sensor_triggered:
                time.sleep(0.2)
                continue

            last_capture = now

            # grab frame
            if ON_PI:
                raw_capture.truncate(0)  # clear buffer
                frame = next(stream_iter).array
            else:
                ret, frame = cap.read()
                if not ret:
                    print("Camera read failed, retrying...")
                    time.sleep(1)
                    continue

            # Run detection pipeline
            try:
                result = detect_and_read_plate_from_image(frame)
            except Exception as e:
                print("Pipeline error:", e)
                traceback.print_exc()
                result = None

            if result is None:
                print("No plate detected in frame.")
                # turn LEDs off to show "no result"
                set_led(green_on=False, red_on=False)
                continue

            # Save annotated locally
            os.makedirs(UPLOAD_OUTPUT_DIR, exist_ok=True)
            filename = f"scan_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}.jpg"
            local_path = os.path.join(UPLOAD_OUTPUT_DIR, filename)
            cv2.imwrite(local_path, result["annotated_image"])
            print(f"Saved local annotated image: {local_path}")

            # Upload to Firebase Storage
            remote_path = f"scans/{filename}"
            try:
                image_url = upload_image_to_storage(bucket, result["annotated_image"], remote_path)
            except Exception as e:
                print("Upload error:", e)
                image_url = None

            # Build scan document
            scan_doc = {
                "id": uuid.uuid4().hex,
                "licensePlate": result.get("plate"),
                "color": result.get("color"),
                "colorConf": result.get("color_conf", 0.0),
                "make_model": result.get("make_model"),
                "mm_conf": result.get("mm_conf", 0.0),
                "timestamp": firestore.SERVER_TIMESTAMP,
                "deviceId": "raspberry_pi_01",
                "scanSource": "raspberry_pi_camera",
                "imageUrl": image_url,
            }

            # Write scan doc to scans collection
            try:
                write_scan_to_firestore(db, scan_doc)
                print("Scan document written to Firestore")
            except Exception as e:
                print("Firestore write error (scans):", e)
                traceback.print_exc()

            # Query Firebase to determine verification
            try:
                verified = verify_plate_with_firebase(db, scan_doc)
            except Exception as e:
                print("Error during firebase verification check:", e)
                traceback.print_exc()
                verified = False

            # Update vehicle document with scan & verification info
            try:
                upsert_vehicle_from_scan(db, scan_doc, verified=verified)
            except Exception as e:
                print("Firestore write error (vehicles):", e)
                traceback.print_exc()

            # Set LEDs according to verification result from Firebase
            set_led(green_on=bool(verified), red_on=not bool(verified))
            print(f"Verification result for plate {scan_doc.get('licensePlate')}: {verified}")

            # small delay before next capture
            time.sleep(1)

    except KeyboardInterrupt:
        print("Shutting down capture loop...")
    finally:
        if ON_PI:
            GPIO.cleanup()
        else:
            try:
                cap.release()
            except Exception:
                pass

# ---------------- Main ----------------
def main():
    print("Initializing...")
    db, bucket = init_firebase()
    setup_gpio()
    capture_loop(db, bucket)

if __name__ == "__main__":
    main()
