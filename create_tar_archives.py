import os
import tarfile
from pathlib import Path
import argparse

# max tar size (500 MiB each)
MAX_TAR_SIZE = 500 * 1024 * 1024 

def create_archives(source_dir, target_dir, split_name):
    """
    Creates sharded .tar archives from .npz files in a source directory.

    Args:
        source_dir (Path): The directory containing 'sample_*.npz' files.
        target_dir (Path): The directory where 'split_name_XXX.tar' files will be saved.
        split_name (str): The name of the split (e.g., 'train' or 'val').
    """
    print(f"Processing {split_name} data from: {source_dir}")
    
    target_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(list(source_dir.glob("sample_*.npz")))
    
    if not npz_files:
        print(f"Warning: No 'sample_*.npz' files found in {source_dir}")
        return

    tar_index = 0
    current_tar_size = 0
    tar_path = target_dir / f"{split_name}_{tar_index:03d}.tar"
    tar = tarfile.open(tar_path, 'w')
    
    print(f"Creating new archive: {tar_path.name}")

    for npz_file in npz_files:
        file_size = os.path.getsize(npz_file)
        

        if current_tar_size + file_size > MAX_TAR_SIZE and current_tar_size > 0:
            tar.close()
            print(f"Completed archive: {tar_path.name} (Size: {current_tar_size / (1024*1024):.2f} MB)")
            
            tar_index += 1
            current_tar_size = 0
            tar_path = target_dir / f"{split_name}_{tar_index:03d}.tar"
            tar = tarfile.open(tar_path, 'w')
            print(f"Creating new archive: {tar_path.name}")

        # arcname=npz_file.name stores the file without the source_dir path
        tar.add(npz_file, arcname=npz_file.name)
        current_tar_size += file_size
        
    tar.close()
    print(f"Completed archive: {tar_path.name} (Size: {current_tar_size / (1024*1024):.2f} MB)")
    print(f"Finished processing {split_name} split.")

def main():
    parser = argparse.ArgumentParser(description="Compress .npz dataset into sharded .tar files.")
    parser.add_argument(
        "--source_root", 
        type=str, 
        required=True, 
        help="Root directory of the original dataset (containing train/ and val/ with .npz files)"
    )
    parser.add_argument(
        "--target_root", 
        type=str, 
        default="dataset_tars", 
        help="Root directory to save the new tar archives (will create train/ and val/ subfolders)"
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    target_root = Path(args.target_root)

    # Process the training split
    create_archives(
        source_dir=source_root / "train",
        target_dir=target_root / "train",
        split_name="train"
    )
    
    # Process the validation split
    create_archives(
        source_dir=source_root / "val",
        target_dir=target_root / "val",
        split_name="val"
    )

    print("\nAll done. Tar archives are ready in:")
    print(f"Train: {target_root / 'train'}")
    print(f"Val:   {target_root / 'val'}")

if __name__ == "__main__":
    main()