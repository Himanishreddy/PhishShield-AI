import shutil
from pathlib import Path

PATH_LIST = Path(
    r"C:\Users\vrlre\Phishing\data_collection\enron_ood_v3_paths.txt"
)

DST = Path(
    r"C:\Users\vrlre\Phishing\data_collection\enron_ood_v3_sample"
)

DST.mkdir(parents=True, exist_ok=True)

files = [
    x.strip()
    for x in PATH_LIST.read_text(encoding="ascii").splitlines()
    if x.strip()
]

print("Found:", len(files))

copied = 0

for i, path in enumerate(files, 1):
    try:
        source = "\\\\?\\" + path
        destination = DST / f"email_{i:04d}.eml"

        shutil.copyfile(source, destination)
        copied += 1

    except Exception as e:
        print("FAILED:", path)
        print(e)

print("Copied:", copied)