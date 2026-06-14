import os
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
COMBINED_DIR = os.path.join(DATA_DIR, 'combined')

SOURCES = [
    os.path.join(DATA_DIR, 'PlantVillage', 'PlantVillage'),
    os.path.join(DATA_DIR, 'rice', 'Rice_Leaf_AUG'),
    os.path.join(DATA_DIR, 'wheat', 'Large Wheat Disease Classification Dataset'),
    os.path.join(DATA_DIR, 'cotton', 'Cotton Disease', 'train'),
]

def prepare():
    print("Creating combined dataset...")
    os.makedirs(COMBINED_DIR, exist_ok=True)
    
    for source in SOURCES:
        if not os.path.exists(source):
            print(f"Skipping {source} - not found")
            continue
        
        for class_name in os.listdir(source):
            class_path = os.path.join(source, class_name)
            if not os.path.isdir(class_path):
                continue
            
            dest_path = os.path.join(COMBINED_DIR, class_name)
            os.makedirs(dest_path, exist_ok=True)
            
            for image in os.listdir(class_path):
                src_file = os.path.join(class_path, image)
                dst_file = os.path.join(dest_path, image)
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
        
        print(f"Done: {source}")
    
    classes = os.listdir(COMBINED_DIR)
    print(f"\nTotal classes: {len(classes)}")
    print("Classes:", classes)

if __name__ == '__main__':
    prepare()