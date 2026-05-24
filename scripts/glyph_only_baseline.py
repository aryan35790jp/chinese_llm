"""
glyph_only_baseline.py — pure-vision encoder with NO language training.

For each character in the dataset:
    1. Render at 96×96 in Noto Sans CJK SC (or any system CJK font)
    2. Encode through a frozen ImageNet ResNet-18, take penultimate
       activations (512-d)
    3. Cache as if it were just another model:
       cache/embeddings/glyph_only__resnet18/layer00_char.npy

This gives us a clean form-only signal with zero distributional or
linguistic input. If radical cohesion is *strong* here but *vanishes*
under semantic control on contextual models, we have isolated the
orthographic channel cleanly and we can quote that decomposition in the
paper.

We treat this as a pseudo-model so that all downstream scripts
(layer_wise_analysis, expanded_semantic_control, probing_classifier)
include it for free.

Caveats:
    - Without a CJK font installed, rendering will produce empty boxes.
      We try common font paths (Linux/macOS/Windows) and warn loudly if
      none are usable.
    - ResNet-18 sees a 96×96 character image, scaled to 224×224. This
      is a deliberately simple baseline; tuning the encoder would
      defeat the purpose.

Output:
    cache/embeddings/glyph_only__resnet18/layer00_char.npy   (N×512)
    cache/embeddings/glyph_only__resnet18/layer00_mean.npy   (= same)
    cache/embeddings/glyph_only__resnet18/layer00_cls.npy    (= same)
    cache/embeddings/glyph_only__resnet18/charlist.txt
    cache/embeddings_iso/glyph_only__resnet18/layer00_*.npy  (corrected)

Runtime: 5–10 minutes on GPU, ~30 minutes on CPU. RAM: <2 GB.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radical_lib import (  # noqa: E402
    CACHE_DIR,
    set_seed,
    get_device,
    load_radical_dataset,
    save_layer_embeddings,
    fit_isotropy,
    apply_isotropy,
)
from radical_lib.embeddings import model_dir, model_tag  # noqa: E402
from scripts.new.config import ISOTROPY_K  # noqa: E402

set_seed()

PSEUDO_MODEL_ID = "glyph_only/resnet18"
ISO_DIR = CACHE_DIR / "embeddings_iso"


def find_cjk_font() -> Optional[str]:
    """Try to find a CJK-capable .ttf/.otf on the system."""
    candidates = [
        # Linux
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
        "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        # Windows
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\Deng.ttf",
        r"C:\Windows\Fonts\Dengxian.ttf",
        r"C:\Windows\Fonts\msjh.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def render_char(ch: str, font_path: str, size: int = 96) -> np.ndarray:
    """Render `ch` to a `size × size` grayscale numpy array, value in [0, 1]."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, int(size * 0.85))
    # measure
    try:
        bbox = draw.textbbox((0, 0), ch, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (size - w) // 2 - bbox[0]
        y = (size - h) // 2 - bbox[1]
    except AttributeError:  # very old PIL
        w, h = draw.textsize(ch, font=font)
        x, y = (size - w) // 2, (size - h) // 2
    draw.text((x, y), ch, fill=0, font=font)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def main():
    font = find_cjk_font()
    if font is None:
        print("[fatal] no CJK font found. Install Noto Sans CJK or set the path manually.")
        sys.exit(1)
    print(f"Using CJK font: {font}")

    df = load_radical_dataset()
    chars: List[str] = df["char"].tolist()
    print(f"Rendering {len(chars)} characters …")

    import torch
    from torchvision import models, transforms

    device = get_device()
    print(f"Device: {device}")
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Strip the classifier: keep up to global avgpool → 512-d
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval().to(device)

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),  # ResNet expects 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    feats = np.zeros((len(chars), 512), dtype=np.float32)
    bs = 32
    with torch.no_grad():
        for i in tqdm(range(0, len(chars), bs)):
            batch = []
            for c in chars[i:i + bs]:
                arr = render_char(c, font)  # H×W in [0,1]
                # convert grayscale array → uint8 image for ToPILImage
                img = (arr * 255).astype(np.uint8)
                batch.append(preprocess(img))
            x = torch.stack(batch).to(device)
            f = resnet(x).squeeze(-1).squeeze(-1)  # B × 512
            feats[i:i + len(batch)] = f.detach().cpu().numpy().astype(np.float32)

    # Save under the standard cache layout — exactly one "layer" (0).
    save_layer_embeddings(PSEUDO_MODEL_ID, layer=0, embeddings=feats, chars=chars, pool="char")
    save_layer_embeddings(PSEUDO_MODEL_ID, layer=0, embeddings=feats, chars=chars, pool="mean")
    save_layer_embeddings(PSEUDO_MODEL_ID, layer=0, embeddings=feats, chars=chars, pool="cls")
    print(f"Cached vision features → {model_dir(PSEUDO_MODEL_ID)}")

    # Also fit isotropy and save corrected version (so layer_wise sees it).
    iso_dir = ISO_DIR / model_tag(PSEUDO_MODEL_ID)
    iso_dir.mkdir(parents=True, exist_ok=True)
    params = fit_isotropy(feats, k=ISOTROPY_K)
    Xc = apply_isotropy(feats, params).astype(np.float32)
    for pool in ("char", "mean", "cls"):
        np.save(iso_dir / f"layer00_{pool}.npy", Xc)
    print(f"Cached isotropy-corrected vision features → {iso_dir}")


if __name__ == "__main__":
    main()
