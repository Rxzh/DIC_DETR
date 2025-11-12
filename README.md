
```bash
# Run the script, pointing to the root of your *original* dataset
# (e.g., the folder containing the 'train' and 'val' .npz subfolders)
python create_tar_archives.py --source_root radon_dataset --target_root radon_dataset_tars

# After running, you will have a new folder:
#
# dataset_tars/
# ├── train/
# │   ├── train_000.tar
# │   ├── train_001.tar
# │   └── ...
# └── val/
#     ├── val_000.tar
#     └── ...
```