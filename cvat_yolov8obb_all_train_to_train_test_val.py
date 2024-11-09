import os
import random
import shutil
import yaml

# Define input and output directories
INPUT_DIR = "C:/Users/Studio/Documents/GitHub/dice-reader/data/dice-reader_test_set"
OUTPUT_DIR = INPUT_DIR + "_cov"
TRAIN = .4
TEST= .4
VAL = .2

def main():
    if TRAIN + TEST + VAL != 1:
        print("percent distrabution off, critical error, exiting")
        exit(1) #distrabution off
    copy_full_dir()
    mod_dir()

def mod_dir():
    my_yaml = None
    with open(OUTPUT_DIR+"/data.yaml") as stream:
        try:
            my_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    if os.path.exists(OUTPUT_DIR+"/data.yaml"):
            os.remove(OUTPUT_DIR+"/data.yaml")
    my_yaml['path']=OUTPUT_DIR
    my_yaml['train'] = 'images/train'
    my_yaml['test'] = 'images/test'
    my_yaml['val'] = 'images/val'
    with open(OUTPUT_DIR+'/data.yaml', 'w') as yaml_file:
        yaml.dump(my_yaml, yaml_file, default_flow_style=False)
    os.makedirs(OUTPUT_DIR+"/labels/test", exist_ok=True)
    os.makedirs(OUTPUT_DIR+"/labels/val", exist_ok=True)
    os.makedirs(OUTPUT_DIR+"/images/test", exist_ok=True)
    os.makedirs(OUTPUT_DIR+"/images/val", exist_ok=True)
    if os.path.exists(OUTPUT_DIR+"/train.txt"):
            os.remove(OUTPUT_DIR+"/train.txt")

    

    ls = os.listdir(OUTPUT_DIR+"/images/train/")
    extra_dir = None
    for item in ls:
        if "." not in item:
            extra_dir = item
    if extra_dir is not None:
        allimagefiles = os.listdir(OUTPUT_DIR+"/images/train/"+extra_dir)
        alllabelfiles = os.listdir(OUTPUT_DIR+"/labels/train/"+extra_dir)

        test_and_val_image_files = random.sample(allimagefiles, int(len(allimagefiles)*TEST)+int(len(allimagefiles)*VAL))
        val_image_files = random.sample(test_and_val_image_files, int(len(allimagefiles)*VAL))
        
        test_and_val_label_files = []
        val_label_files = []
        for item in test_and_val_image_files:
            test_and_val_label_files.append(item.replace(item.split(".")[1], "txt"))
        for item in val_image_files:
            val_label_files.append(item.replace(item.split(".")[1], "txt"))

        for f in allimagefiles:
            if f in val_image_files:
                shutil.move(OUTPUT_DIR+"/images/train/"+extra_dir+"/"+f, OUTPUT_DIR+"/images/val/"+f)
            elif f in test_and_val_image_files:
                shutil.move(OUTPUT_DIR+"/images/train/"+extra_dir+"/"+f, OUTPUT_DIR+"/images/test/"+f)
            else:
                shutil.move(OUTPUT_DIR+"/images/train/"+extra_dir+"/"+f, OUTPUT_DIR+"/images/train/"+f)
        if os.path.exists(OUTPUT_DIR+"/images/train/"+extra_dir):
            shutil.rmtree(OUTPUT_DIR+"/images/train/"+extra_dir)

        for f in alllabelfiles:
            if f in val_label_files:
                shutil.move(OUTPUT_DIR+"/labels/train/"+extra_dir+"/"+f, OUTPUT_DIR+"/labels/val/"+f)
            elif f in test_and_val_label_files:
                shutil.move(OUTPUT_DIR+"/labels/train/"+extra_dir+"/"+f, OUTPUT_DIR+"/labels/test/"+f)
            else:
                shutil.move(OUTPUT_DIR+"/labels/train/"+extra_dir+"/"+f, OUTPUT_DIR+"/labels/train/"+f)
        if os.path.exists(OUTPUT_DIR+"/labels/train/"+extra_dir):
            shutil.rmtree(OUTPUT_DIR+"/labels/train/"+extra_dir)
    else:
        print("no buffer directory to remove, critical error, exiting...Need to implement handeling")
        exit() #No extra dir

    
def copy_full_dir():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Copying files to: {OUTPUT_DIR}")

    # Walk through each directory and subdirectory
    for root, _, files in os.walk(INPUT_DIR):
        # Create the corresponding directory structure in OUTPUT_DIR
        relative_path = os.path.relpath(root, INPUT_DIR)
        target_dir = os.path.join(OUTPUT_DIR, relative_path)
        os.makedirs(target_dir, exist_ok=True)
        for file in files:
            source_file = os.path.join(root, file)
            destination_file = os.path.join(target_dir, file)
            
            # Copy each file to the new directory structure
            shutil.copy2(source_file, destination_file)  # copy2 preserves metadata
            # print(f"Copied: {source_file} to {destination_file}")

    print("All files and subdirectories copied successfully.")


if __name__ == "__main__":
    main()