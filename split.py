import os
import shutil
import random
from pathlib import Path

def split_dataset(input_dir, output_dir, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1):

    os.makedirs(output_dir)

    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    val_dir = os.path.join(output_dir, 'val')
    
    for split_dir in [train_dir, test_dir, val_dir]:
        os.makedirs(split_dir, exist_ok=True)
    
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
            
        # Get all files in the class directory
        files = [f for f in os.listdir(class_dir) if not f.startswith('.')]
        random.shuffle(files)
        
        # Calculate split indices
        num_files = len(files)
        train_end = int(num_files * train_ratio)
        test_end = train_end + int(num_files * test_ratio)
        
        # Split files
        train_files = files[:train_end]
        test_files = files[train_end:test_end]
        val_files = files[test_end:]
        
        # Create class directories in each split and copy files
        for split, split_files in [('train', train_files), ('test', test_files), ('val', val_files)]:
            class_split_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(class_split_dir, exist_ok=True)
            
            for file in split_files:
                src = os.path.join(class_dir, file)
                dst = os.path.join(class_split_dir, file)
                shutil.copy2(src, dst)
                
        print(f"Processed {class_name}: {len(train_files)} train, {len(test_files)} test, {len(val_files)} val")

if __name__ == "__main__":
    input_directory = "./datasets/fishnet/Image_Library"
    output_directory = "./dataset_split"
    
    split_dataset(input_directory, output_directory)