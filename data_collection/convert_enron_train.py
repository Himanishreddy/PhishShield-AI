import csv
from pathlib import Path
from email import policy
from email.parser import BytesParser

SRC = Path(r".\data_collection\enron_train_sample")
OUT = Path(r".\data_collection\enron_ham_train_2000.csv")

def extract_text(msg):
    parts = []

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    parts.append(part.get_content())
                except Exception:
                    pass
    else:
        if msg.get_content_type() == "text/plain":
            try:
                parts.append(msg.get_content())
            except Exception:
                pass

    return "\n".join(parts).strip()

rows = []

files = list(SRC.glob("*.eml"))
print(f"Input files: {len(files)}")

for path in files:
    try:
        with open(path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)

        text = extract_text(msg)

        if not text:
            continue

        rows.append({
            "Email Text": text,
            "Email Type": "ham",
            "Source": "enron"
        })

    except Exception as e:
        print(f"Failed: {path.name} -> {e}")

with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["Email Text", "Email Type", "Source"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Valid emails: {len(rows)}")
print(f"Saved to: {OUT}")