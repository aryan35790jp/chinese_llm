"""
Build radical map from Unihan database.
Extract character → Kangxi radical mapping from kRSUnicode field.
"""

radical_map = {}

with open("data/unihan/Unihan_IRGSources.txt", encoding="utf-8") as f:
    for line in f:
        if line.startswith("U+") and "kRSUnicode" in line:
            parts = line.strip().split("\t")
            codepoint = parts[0]
            radical_info = parts[2]  # e.g. "85.5"

            char = chr(int(codepoint[2:], 16))
            radical = radical_info.split(".")[0].rstrip("'")

            if char not in radical_map:
                radical_map[char] = int(radical)

print(f"Total characters with radical info: {len(radical_map)}")

# Show sample
samples = list(radical_map.items())[:20]
for char, rad in samples:
    print(f"  {char} → radical {rad}")
