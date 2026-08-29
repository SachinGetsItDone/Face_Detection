import os
import cv2
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'database')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load AI Models
yolo_model_path = os.path.join(os.getcwd(), "model", "model.pt")
print("Loading YOLOv8 Detector...")
detector = YOLO(yolo_model_path) if os.path.exists(yolo_model_path) else None

print("Loading PyTorch FaceNet...")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

db_embeddings = {}
global_latest_frame = None

# FaceNet Embedding Function
def get_face_embedding(resnet, face_crop, device):
    try:
        face_resized = cv2.resize(face_crop, (160, 160))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(face_rgb).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet(tensor).cpu().numpy().flatten()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        return embedding
    except Exception:
        return None

# Database Loading
def load_database():
    global db_embeddings
    new_embeddings = {}
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    for file_name in os.listdir(app.config['UPLOAD_FOLDER']):
        if file_name.lower().endswith(valid_exts):
            name = os.path.splitext(file_name)[0]
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            img = cv2.imread(img_path)
            if img is None: continue
            
            face_crop = img
            if detector is not None:
                results = detector(img, verbose=False)
                if len(results[0].boxes) > 0:
                    box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box
                    h_img, w_img, _ = img.shape
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    face_crop = img[y1:y2, x1:x2]
                    
            emb = get_face_embedding(resnet, face_crop, device)
            if emb is not None:
                new_embeddings[name] = emb
                
    db_embeddings = new_embeddings

load_database()

# MJPEG Video Generator
def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam for the video feed.")
        return
    try:
        # Delegate frame production; the finally guarantees the camera is
        # released when the client disconnects (GeneratorExit) or an error occurs.
        yield from _stream_frames(cap)
    finally:
        cap.release()


def _stream_frames(cap):
    global global_latest_frame
    frame_count = 0
    cached_results = []
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        global_latest_frame = frame.copy()
        
        frame_count += 1
        
        if frame_count % 3 == 0 and detector is not None:
            cached_results = []
            results = detector(frame, verbose=False)
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                for box, conf in zip(boxes, confs):
                    if conf < 0.45: continue
                    x1, y1, x2, y2 = box.astype(int)
                    h, w, _ = frame.shape
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    face_crop = frame[y1:y2, x1:x2]
                    
                    identity = "Unknown"
                    if len(db_embeddings) > 0:
                        emb = get_face_embedding(resnet, face_crop, device)
                        if emb is not None:
                            best_match = "Unknown"
                            min_dist = 1.0
                            for name, ref_emb in db_embeddings.items():
                                dist = 1.0 - float(np.dot(emb, ref_emb))
                                if dist < min_dist and dist < 0.42:
                                    min_dist = dist
                                    best_match = name
                            identity = best_match
                        
                    cached_results.append({
                        "box": (x1, y1, x2, y2),
                        "identity": identity
                    })
                    
        for res in cached_results:
            x1, y1, x2, y2 = res["box"]
            identity = res["identity"]
            
            if identity != "Unknown":
                color, label = (0, 255, 0), f"{identity}"
            else:
                color, label = (0, 255, 255), "Unknown"
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ROUTES
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({"error": "No file or name provided"}), 400
    
    file = request.files['image']
    name = request.form['name'].strip()
    if file.filename == '' or not name:
        return jsonify({"error": "Empty file or name"}), 400
        
    ext = os.path.splitext(file.filename)[1]
    filename = secure_filename(f"{name}{ext}")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    load_database() # Refresh DB embeddings
    return jsonify({"message": f"Successfully enrolled {name}"}), 200

@app.route('/api/live_enroll', methods=['POST'])
def live_enroll():
    global global_latest_frame
    data = request.get_json()
    if not data or 'name' not in data:
        return jsonify({"error": "No name provided"}), 400
        
    name = data['name'].strip()
    if not name:
        return jsonify({"error": "Empty name"}), 400
        
    if global_latest_frame is None:
        return jsonify({"error": "Camera feed not ready"}), 400
        
    # Check if a face is actually present using the loaded YOLO detector
    if detector is not None:
        results = detector(global_latest_frame, verbose=False)
        if len(results[0].boxes) == 0:
            return jsonify({"error": "No face detected in the live frame! Please look at the camera."}), 400
        
    filename = secure_filename(f"{name}.jpg")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cv2.imwrite(file_path, global_latest_frame)
    
    load_database() # Refresh DB embeddings
    return jsonify({"message": f"Successfully enrolled {name} from camera"}), 200

@app.route('/api/faces', methods=['GET'])
def list_faces():
    faces = []
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for file_name in os.listdir(app.config['UPLOAD_FOLDER']):
            if file_name.lower().endswith(valid_exts):
                faces.append({
                    "name": os.path.splitext(file_name)[0],
                    "filename": file_name,
                    "url": f"/database/{file_name}"
                })
    return jsonify(faces)

@app.route('/database/<filename>')
def serve_db_image(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/faces/<filename>', methods=['DELETE'])
def delete_face(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if os.path.exists(file_path):
        os.remove(file_path)
        load_database() # Refresh
        return jsonify({"message": "Deleted successfully"}), 200
    return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    # use_reloader=False prevents Werkzeug from importing this module twice,
    # which would load the YOLO/FaceNet/ONNX models (and grab the camera) twice.
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
