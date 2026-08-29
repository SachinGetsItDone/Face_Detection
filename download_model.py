from huggingface_hub import snapshot_download
import os

def download_model():
    # Define the local directory where you want to save the model
    # This will create a 'model' folder inside your current directory (Face_detection)
    download_dir = os.path.join(os.getcwd(), "model")
    
    print(f"Downloading YOLOv8 Face Detection model to {download_dir}...")
    
    # Download the entire model repository
    model_path = snapshot_download(
        repo_id="arnabdhar/YOLOv8-Face-Detection",
        local_dir=download_dir,
    )
    
    print(f"✅ Model downloaded successfully!")
    print(f"You can find the files at: {model_path}")

if __name__ == "__main__":
    download_model()
