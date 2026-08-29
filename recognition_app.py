import cv2
import os
import sys
import torch
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

def get_face_embedding(resnet, face_crop, device):
    """Computes a 512-dimensional face embedding using PyTorch FaceNet."""
    try:
        face_resized = cv2.resize(face_crop, (160, 160))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(face_rgb).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = resnet(tensor).cpu().numpy().flatten()
            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        return embedding
    except Exception as e:
        return None

def load_database(resnet, yolo, db_path, device):
    """Loads reference face embeddings from the database folder."""
    db_embeddings = {}
    if not os.path.exists(db_path):
        os.makedirs(db_path, exist_ok=True)
        return db_embeddings

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    for file_name in os.listdir(db_path):
        if file_name.lower().endswith(valid_exts):
            name = os.path.splitext(file_name)[0]
            img_path = os.path.join(db_path, file_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Detect face in reference photo
            face_crop = img
            if yolo is not None:
                results = yolo(img, verbose=False)
                if len(results[0].boxes) > 0:
                    box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box
                    h_img, w_img, _ = img.shape
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    face_crop = img[y1:y2, x1:x2]

            emb = get_face_embedding(resnet, face_crop, device)
            if emb is not None:
                db_embeddings[name] = emb
                print(f"Loaded reference face for: '{name}'")

    return db_embeddings

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths
    yolo_model_path = os.path.join(os.getcwd(), "model", "model.pt")
    db_path = os.path.join(os.getcwd(), "database")

    if not os.path.exists(yolo_model_path):
        print(f"Error: YOLO model file not found at '{yolo_model_path}'. Please run download_model.py first.")
        sys.exit(1)

    print("Loading YOLOv8 Face Detector...")
    yolo = YOLO(yolo_model_path)

    print("Loading PyTorch FaceNet Recognition Model...")
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    print("\nLoading known faces from database...")
    db_embeddings = load_database(resnet, yolo, db_path, device)
    print(f"Total reference faces loaded: {len(db_embeddings)}\n")

    print("==================================================")
    print("      Face Recognition App        ")
    print("==================================================")
    print("Press 'q' in the camera window to exit.")
    print("==================================================\n")

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    # Frame sampling rate for integrated GPU performance
    FRAME_INTERVAL = 3
    frame_count = 0
    cached_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_count += 1

        if frame_count % FRAME_INTERVAL == 0:
            cached_results = []
            results = yolo(frame, verbose=False)

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

                for box, conf in zip(boxes, confs):
                    if conf < 0.45:
                        continue

                    x1, y1, x2, y2 = box.astype(int)
                    h_img, w_img, _ = frame.shape
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)

                    face_crop = frame[y1:y2, x1:x2]

                    # Face Recognition against Database
                    identity = "Unknown"
                    if len(db_embeddings) > 0:
                        emb = get_face_embedding(resnet, face_crop, device)
                        if emb is not None:
                            best_match = "Unknown"
                            min_dist = 1.0  # Cosine distance threshold (lower is closer match)

                            for name, ref_emb in db_embeddings.items():
                                dist = 1.0 - float(np.dot(emb, ref_emb))
                                if dist < min_dist and dist < 0.42:  # Threshold for match
                                    min_dist = dist
                                    best_match = name

                            identity = best_match

                    cached_results.append({
                        "box": (x1, y1, x2, y2),
                        "identity": identity
                    })

        # Draw overlays
        for res in cached_results:
            x1, y1, x2, y2 = res["box"]
            identity = res["identity"]

            if identity != "Unknown":
                color, label = (0, 255, 0), f"{identity}"
            else:
                color, label = (0, 255, 255), "Unknown"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, "Press 'q' to exit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Face Recognition Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
