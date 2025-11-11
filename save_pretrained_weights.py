from transformers import DetrImageProcessor, DetrForObjectDetection
from pathlib import Path


MODEL_ID = "facebook/detr-resnet-50"
REVISION = "no_timm"

SAVE_DIRECTORY = Path("pretrained_weights/facebook-detr-resnet-50")
# --- End Configuration ---

if __name__ == "__main__":
    
    # Ensure the directory exists
    SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading model '{MODEL_ID}' (revision: {REVISION})...")
    
    # 1. Download and load the processor
    processor = DetrImageProcessor.from_pretrained(
        MODEL_ID, 
        revision=REVISION
    )
    
    # 2. Download and load the model
    model = DetrForObjectDetection.from_pretrained(
        MODEL_ID, 
        revision=REVISION
    )
    
    print("Download complete. Saving to local directory...")
    
    # 3. Save all files to the specified directory
    processor.save_pretrained(SAVE_DIRECTORY)
    model.save_pretrained(SAVE_DIRECTORY)
    
    print(f"Successfully saved model and processor to:")
    print(f"{SAVE_DIRECTORY.resolve()}")
    print("\nYou can now use this path in your main script.")