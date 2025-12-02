from ultralytics import YOLO
from pathlib import Path
import zipfile
import sys


def prepare_dataset(root: Path, zip_name: str, extract_dir: str = "dataset") -> Path:
    """
    - Controlla che lo zip esista
    - Lo estrae (solo la prima volta)
    - Cerca data.yaml dentro la cartella estratta
    - Ritorna il path completo al data.yaml
    """
    zip_path = root / zip_name
    out_dir = root / extract_dir

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip dataset non trovato: {zip_path}")

    if not out_dir.exists():
        print(f"[INFO] Estraggo {zip_path} in {out_dir} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
    else:
        print(f"[INFO] Cartella dataset già esistente: {out_dir}")

    # Cerca data.yaml dentro la cartella estratta (ricorsivamente)
    data_yamls = list(out_dir.rglob("data.yaml"))
    if not data_yamls:
        raise FileNotFoundError(f"Nessun data.yaml trovato in {out_dir}")
    if len(data_yamls) > 1:
        print("[WARN] Trovati più data.yaml, uso il primo:", data_yamls[0])

    return data_yamls[0]


def main():
    # Cartella in cui si trova questo script
    root = Path(__file__).resolve().parent

    # Nome dello zip come nel tuo progetto
    dataset_zip_name = "Distributed.v2i.yolov11.zip"

    # Prepara dataset (unzippa + trova data.yaml)
    try:
        data_yaml_path = prepare_dataset(root, dataset_zip_name)
    except FileNotFoundError as e:
        print(f"[ERRORE] {e}")
        sys.exit(1)

    print(f"[INFO] Uso data.yaml: {data_yaml_path}")

    # Percorso al modello YOLOv11s (assicurati che il nome combaci)
    model_path = root / "yolo11s.pt"  # cambia in "yolov11s.pt" se il file si chiama così

    if not model_path.exists():
        print(f"[ERRORE] File del modello non trovato: {model_path}")
        sys.exit(1)

    print(f"[INFO] Uso il modello: {model_path}")

    # Carica il modello YOLO
    model = YOLO(str(model_path))

    # Allena il modello usando il data.yaml trovato
    results = model.train(
        data=str(data_yaml_path),
        epochs=200,
        imgsz=640,
        batch=16,
        device=0,   # 0 = GPU
        workers=4,
        val=True,
    )

    # Validazione finale
    model.val()


if __name__ == "__main__":
    main()
