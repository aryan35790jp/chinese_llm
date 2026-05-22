"""
Prepare the final dataset:
1. Parse radical map from Unihan (kRSUnicode)
2. Filter to common Chinese characters using BERT tokenizer vocab as frequency proxy
3. Keep only radicals with >= 20 characters
4. Save as CSV
"""
import pandas as pd
from transformers import AutoTokenizer

# ═══════════════════════════════════════════
# Step 1: Build radical map from Unihan
# ═══════════════════════════════════════════
radical_map = {}
with open("data/unihan/Unihan_IRGSources.txt", encoding="utf-8") as f:
    for line in f:
        if line.startswith("U+") and "kRSUnicode" in line:
            parts = line.strip().split("\t")
            codepoint = parts[0]
            radical_info = parts[2]
            char = chr(int(codepoint[2:], 16))
            radical = radical_info.split(".")[0].rstrip("'")
            if char not in radical_map:
                radical_map[char] = int(radical)

print(f"Unihan total: {len(radical_map)} characters")

# ═══════════════════════════════════════════
# Step 2: Filter to common Chinese characters
# Use Chinese-BERT tokenizer vocab as frequency proxy —
# characters in the vocab are commonly used
# ═══════════════════════════════════════════
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
vocab = set(tokenizer.get_vocab().keys())

# Keep single CJK characters that are in both Unihan radical map AND tokenizer vocab
common_radical_map = {}
for char, radical in radical_map.items():
    cp = ord(char)
    if 0x4E00 <= cp <= 0x9FFF and char in vocab:
        common_radical_map[char] = radical

print(f"After tokenizer vocab filter: {len(common_radical_map)} characters")

# ═══════════════════════════════════════════
# Step 3: Build dataframe
# ═══════════════════════════════════════════
rows = []
for char, radical in common_radical_map.items():
    rows.append({"char": char, "radical": radical})

df = pd.DataFrame(rows)
print(f"\nDataFrame shape: {df.shape}")

# ═══════════════════════════════════════════
# Step 4: Keep only radicals with >= 20 characters
# ═══════════════════════════════════════════
radical_counts = df["radical"].value_counts()
valid_radicals = radical_counts[radical_counts >= 20].index
df_final = df[df["radical"].isin(valid_radicals)].copy()

print(f"\nRadicals with >= 20 chars: {len(valid_radicals)} radicals")
print(f"Characters after radical filter: {len(df_final)}")

print(f"\n{'='*50}")
print(f"FINAL DATASET")
print(f"{'='*50}")
print(f"Characters: {len(df_final)}")
print(f"Radicals:   {df_final['radical'].nunique()}")
print(f"\nTop 20 radicals by count:")
print(df_final["radical"].value_counts().head(20))

# ═══════════════════════════════════════════
# Save
# ═══════════════════════════════════════════
df_final.to_csv("data/radical_dataset.csv", index=False)
print(f"\nSaved to data/radical_dataset.csv")

# Also save a summary
summary = df_final.groupby("radical").agg(
    count=("char", "count"),
    sample_chars=("char", lambda x: "".join(x.head(5)))
).sort_values("count", ascending=False)
summary.to_csv("data/radical_summary.csv")
print("Saved radical summary to data/radical_summary.csv")
print(f"\nSample radicals:")
print(summary.head(20).to_string())
