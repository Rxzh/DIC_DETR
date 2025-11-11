from pathlib import Path

class Config:

    # --- Data Generation Params ---
    # These are used by generate_dataset.py
    IMG_SIZE = 256           # Corresponds to 'size' (rho dimension)
    THETA_RES = 256          # Number of angles (theta dimension)
    MIN_LINES = 10           # Minimum number of lines per sample
    MAX_LINES = 100           # Maximum number of lines per sample
    
    # --- Dataset Size ---
    # Total number of samples to create
    TOTAL_TRAIN_SAMPLES = 50000 
    TOTAL_VAL_SAMPLES = 5000
    
    # --- File/Directory Paths ---
    # Base directory to store the .npz files
    DATA_DIR = Path("radon_dataset") 
    
    # --- Training/Model Params ---
    # These are used by train.py
    BATCH_SIZE = 8
    NUM_WORKERS = 8
    
    # Box definitions for DETR
    BOX_WIDTH_PX = 9         
    BOX_HEIGHT_PX = 9
    
    # Model/Processor
    PROCESSOR_NAME = "facebook/detr-resnet-50"
    
    # Our single class: "peak"
    CLASS_ID = 0 

# Make the config object easily importable
config = Config()