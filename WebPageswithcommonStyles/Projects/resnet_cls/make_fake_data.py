import numpy as np
from pathlib import Path
from PIL import Image

rng = np.random.default_rng(0)
for cls in ["Normal", "Pneumonia", "COVID-19"]:
    folder = Path("raw") / cls
    folder.mkdir(parents=True, exist_ok=True)
    for patient in range(15):                    # 15 patients per class
        for k in range(2):                       # 2 images per patient
            img = rng.integers(0, 256, (256, 256), dtype=np.uint8)
            Image.fromarray(img).save(folder / f"{cls[:3]}{patient:02d}_{k}.png")
