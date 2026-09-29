import random
import pandas as pd
from email import policy
from email.parser import BytesParser

FILELIST = r"C:\Users\vrlre\Phishing\data_collection\enron_filelist.txt"
OUTPUT = r"C:\Users\vrlre\Phishing\data_collection\external_ham_500.csv"

TARGET = 500
SEED = 42

random.seed(SEED)

with open(FILELIST, "r", encoding="utf-8") as f:
    files = [line.strip() for line in f if line.strip()]

print(f"Found {len(files)} email files.")

random.shuffle(files)


def extract_text(msg):
    subject = msg.get("Subject", "") or ""
    body_parts = []

    if msg.is_multipart():
        for part in msg.walk():

            if part.get_content_disposition() == "attachment":
                continue

            if part.get_content_type() == "text/plain":
                try:
                    text = part.get_content()
                except Exception:
                    text = ""

                if text:
                    body_parts.append(text)

    else:
        if msg.get_content_type() == "text/plain":
            try:
                body_parts.append(msg.get_content())
            except Exception:
                pass

    body = "\n".join(body_parts)

    return f"{subject}\n{body}".strip()


rows = []
seen = set()
failed = 0

for path in files:

    if len(rows) >= TARGET:
        break

    try:
        with open(path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)

        text = extract_text(msg)

        if not text:
            continue

        normalized = " ".join(text.lower().split())

        if len(normalized) < 50:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)

        rows.append({
            "Email Text": text,
            "Email Type": "Safe Email",
            "source": "enron_external",
            "category": "legitimate"
        })

    except Exception:
        failed += 1


df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8"
)

print()
print("=" * 50)
print("EXTERNAL HAM TEST CREATED")
print("=" * 50)
print(f"Failed files:    {failed}")
print(f"Emails selected: {len(df)}")
print(f"Output: {OUTPUT}")
print("=" * 50)