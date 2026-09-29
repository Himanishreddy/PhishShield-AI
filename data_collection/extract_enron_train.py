from pathlib import Path
import shutil

LIST = Path(r".\data_collection\enron_train_paths.txt")
OUT = Path(r".\data_collection\enron_train_sample")

OUT.mkdir(parents=True, exist_ok=True)

paths = [p.strip() for p in LIST.read_text(encoding="ascii").splitlines() if p.strip()]

print(f"Found: {len(paths)}")

copied = 0

for i, path in enumerate(paths, 1):
    try:
        source = "\\\\?\\" + path
        destination = OUT / f"enron_{i:06d}.eml"

        shutil.copyfile(source, destination)
        copied += 1

        if copied % 100 == 0:
            print(f"Copied: {copied}")

    except Exception as e:
        print(f"Failed: {path} -> {e}")

print(f"Copied: {copied}")