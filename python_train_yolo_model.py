import os
import shutil
import zipfile
import subprocess
from ultralytics import YOLO
import yaml

def create_data_yaml(path_to_classes_txt, path_to_data_yaml, custom_data_dir, data_path):
    # Read classes.txt to get class names
    if not os.path.exists(path_to_classes_txt):
        print(f'classes.txt file not found! Please create a classes.txt labelmap and move it to {path_to_classes_txt}')
        return
    with open(path_to_classes_txt, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    number_of_classes = len(classes)

    # Create data dictionary
    data = {
        'path': data_path,
        'train': 'train/images',
        'val': 'validation/images',
        'nc': number_of_classes,
        'names': classes
    }

    # Write data to YAML file
    with open(path_to_data_yaml, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    print(f'Created config file at {path_to_data_yaml}')

def main():
    # Adesso notebook_dir è il percorso di yolo_pipline_customdata_basic
    notebook_dir = os.path.dirname(os.path.abspath(__file__))

    # Tutto quello che creiamo sta dentro notebook_dir
    custom_data_path = os.path.join(notebook_dir, 'custom_dataset')
    data_path = os.path.join(notebook_dir, 'data')
    zip_file_path = os.path.join(notebook_dir, 'data.zip')
    script_path = os.path.join(notebook_dir, 'train_val_split.py')
    runs_dir = os.path.join(notebook_dir, 'runs') 

    # Se esiste già, lo puliamo
    if os.path.exists(custom_data_path):
        shutil.rmtree(custom_data_path)
    if os.path.exists(runs_dir):
        shutil.rmtree(runs_dir)

    os.makedirs(custom_data_path, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    # Estrai zip in custom_data_path
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(custom_data_path)

    # Lancia split
    subprocess.run(
        ['python3', script_path, '--datapath', custom_data_path, '--train_pct', '0.9'],
        check=True
    )

    print("Dataset preparato con successo!")

    # Path dei file
    data_yaml_path = os.path.join(custom_data_path, 'data.yaml')
    model_path = os.path.join(notebook_dir, 'yolo11s.pt')
    path_to_classes_txt = os.path.join(custom_data_path, 'classes.txt')

    create_data_yaml(path_to_classes_txt, data_yaml_path, custom_data_path, data_path)

    # Stampa file yaml creato
    with open(data_yaml_path, 'r') as f:
        print('\nFile contents:\n')
        print(f.read())

    # Carica modello preaddestrato
    model_1 = YOLO(model_path)

    # Fai training
    model_1.train(
        data=data_yaml_path,
        epochs=60,
        imgsz=640,
        project=runs_dir,  # Imposta il project directory
        name="train"       # Salverà in runs/train/
    )

    print('Training completato!')

    # Percorsi relativi
    validation_images_path = os.path.join(data_path, 'validation', 'images')
    new_model_path = os.path.join(runs_dir, 'train', 'weights', 'best.pt')

    # Ricarica il nuovo modello
    model_2 = YOLO(new_model_path)

    # Predizioni
    model_2.predict(
        source=validation_images_path,
        save=True,
        project=runs_dir,
        name="predictions"
    )

    print('Predizione completata con il nuovo modello!')

if __name__ == "__main__":
    main()
