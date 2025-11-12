from huggingface_hub import HfApi
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Upload dataset tars to Hugging Face Hub")
    parser.add_argument(
        "--repo_id", 
        type=str, 
        required=True, 
        help="Your repo ID on the Hub (e.g., 'username/my-radon-dataset')"
    )
    parser.add_argument(
        "--local_dir", 
        type=str, 
        default="dataset_tars", 
        help="Local directory containing the 'train' and 'val' tar subfolders"
    )
    args = parser.parse_args()

    local_dir_path = Path(args.local_dir)
    if not local_dir_path.exists():
        print(f"Error: Local directory '{args.local_dir}' not found.")
        print("Please run 'create_tar_archives.py' first.")
        return

    print(f"Uploading contents of '{args.local_dir}' to dataset repo '{args.repo_id}'...")
    
    api = HfApi()
    
    api.upload_folder(
        folder_path=args.local_dir,
        repo_id=args.repo_id,
        repo_type="dataset",
        path_in_repo=".", # Upload 'train' and 'val' folders to the root
    )
    
    print("\nUpload complete!")
    print(f"View your dataset at: https://huggingface.co/datasets/{args.repo_id}")

if __name__ == "__main__":
    main()