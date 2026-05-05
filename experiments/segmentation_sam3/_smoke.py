"""One-image smoke test for SAM3 wiring."""
import sys, time, torch
from pathlib import Path
from transformers import Sam3Model, Sam3Processor
from PIL import Image

print("loading model...", flush=True)
t0 = time.time()
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
model.eval()
proc = Sam3Processor.from_pretrained("facebook/sam3")
print(f"loaded in {time.time()-t0:.1f}s, device={device}", flush=True)

img_path = Path(sys.argv[1])
img = Image.open(img_path).convert("RGB")
print("image size:", img.size, flush=True)

t0 = time.time()
inp = proc(images=img, text="light", return_tensors="pt").to(device)
print("input keys:", list(inp.keys()), flush=True)
with torch.no_grad():
    out = model(**inp)
print(f"inference in {time.time()-t0:.1f}s", flush=True)

sizes = inp.get("original_sizes")
target = sizes.tolist() if sizes is not None else [list(img.size[::-1])]
print("target_sizes:", target, flush=True)
res = proc.post_process_instance_segmentation(
    out, threshold=0.5, mask_threshold=0.5, target_sizes=target
)[0]
print("keys:", list(res.keys()), flush=True)
masks = res.get("masks")
print("mask shape:", None if masks is None else (masks.shape, masks.dtype), flush=True)
print("boxes:", res.get("boxes"), flush=True)
print("scores:", res.get("scores"), flush=True)
