import os
import glob
import shutil
import re
import argparse 
# Regex to safely find the number in 'sample_0000001.npz'
INDEX_RE = re.compile(r'sample_(\d{7})\.npz')

def get_max_index(directory):
    """
    Finds the highest sample index in a given directory by
    checking all filenames. Returns -1 if the directory is empty
    or doesn't exist.
    """
    max_idx = -1
    try:
        # Use os.scandir for an efficient directory listing
        for entry in os.scandir(directory):
            if entry.is_file():
                match = INDEX_RE.match(entry.name)
                if match:
                    # Found a valid sample file, get its index
                    idx = int(match.group(1))
                    if idx > max_idx:
                        max_idx = idx
    except FileNotFoundError:
        # Directory doesn't exist yet, which is fine
        pass
    return max_idx

def merge_datasets(src_root, dest_root): # Add parameters
    """
    Moves files from src_root to dest_root, renaming to avoid collisions.
    """
    print(f'Starting merge from "{src_root}" into "{dest_root}"...\n') # Use params
    
    try:
        # Find source subdirs ('train', 'val', etc.)
        for src_dir_entry in os.scandir(src_root): # Use param
            # We only care about subdirectories (train, val)
            if not src_dir_entry.is_dir():
                continue
            
            subfolder_name = src_dir_entry.name
            src_dir = src_dir_entry.path
            dest_dir = os.path.join(dest_root, subfolder_name) # Use param
            
            print(f'--- Processing subfolder: "{subfolder_name}" ---')
            
            # 1. Ensure the destination subfolder (e.g., disk2/radon_dataset/train) exists
            os.makedirs(dest_dir, exist_ok=True)
            
            # 2. Find the highest existing index in the DESTINATION
            max_dest_index = get_max_index(dest_dir)
            next_index = max_dest_index + 1
            print(f'  Destination max index is: {max_dest_index:07d}.')
            print(f'  New files will start at: {next_index:07d}.')
            
            # 3. Get all source files to be moved
            # We sort them to move them in a predictable order
            src_files = sorted(glob.glob(os.path.join(src_dir, 'sample_*.npz')))
            
            if not src_files:
                print(f'  No sample files found in "{src_dir}". Skipping.')
                continue

            # 4. Move and rename each file
            for src_path in src_files:
                new_filename = f'sample_{next_index:07d}.npz'
                dest_path = os.path.join(dest_dir, new_filename)
                
                try:
                    # shutil.move renames and moves. It's atomic if on the same filesystem.
                    shutil.move(src_path, dest_path)
                    print(f'  Moved: {src_path}  ->  {dest_path}')
                    next_index += 1
                except Exception as e:
                    print(f'  [ERROR] Failed to move {src_path}: {e}')
            
            print(f'  Finished processing "{subfolder_name}".\n')

    except FileNotFoundError:
        print(f"[ERROR] Source directory not found: {src_root}") # Use param
        print("Please check your `src_root` path.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        
    print('--- Merge complete ---')

# --- Main execution ---
if __name__ == "__main__":
    # --- Add argument parsing ---
    parser = argparse.ArgumentParser(
        description="Merge dataset samples from a source directory into a destination directory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'src_root', 
        type=str, 
        help='The source dataset root directory (e.g., ./disk1/radon_dataset)'
    )
    parser.add_argument(
        'dest_root', 
        type=str, 
        help='The destination dataset root directory (e.g., ./disk2/radon_dataset)'
    )
    
    args = parser.parse_args()
    # --- End argument parsing ---

    # Before running, please create a backup just in case!
    # print("!!! WARNING: This script will move files permanently. !!!")
    # print("Please back up your data before proceeding.")
    # user_input = input("Type 'MERGE' to confirm and continue: ")
    # if user_input == 'MERGE':
    #     merge_datasets(args.src_root, args.dest_root) # Pass args
    # else:
    #     print("Merge cancelled.")

    # For safety, I've commented out the confirmation.
    # To run the script, you can:
    # 1. Uncomment the lines above to create a safety check.
    # 2. Or, just call merge_datasets() directly below (if you're confident):
    merge_datasets(args.src_root, args.dest_root) # Pass args