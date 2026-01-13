

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from skimage.filters import frangi
from skimage.morphology import (
    skeletonize, remove_small_objects, closing, disk
)


# =========================
# CONFIG
# =========================
IMAGE_PATH = "/Users/geshvad/Desktop/image_s40_i5.png"
MIN_COMPONENT_SIZE = 200        # remove noise blobs
SKELETON_MIN_SIZE = 40          # prune tiny skeleton fragments


# =========================
# 1. LOAD IMAGE
# =========================
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

img = img.astype(np.float32) / 255.0


# =========================
# 2. VESSEL ENHANCEMENT (Frangi tuned for DSA)
# =========================
vesselness = frangi(
    img,
    scale_range=(2, 6),   # DSA vessels are thick
    scale_step=1,
    alpha=0.5,
    beta=0.5,
    gamma=15
)

vesselness = (vesselness - vesselness.min()) / (vesselness.max() + 1e-6)


# =========================
# 3. ADAPTIVE THRESHOLD
# =========================
v_uint8 = (vesselness * 255).astype(np.uint8)

binary = cv2.adaptiveThreshold(
    v_uint8,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    41,      # block size
    -5       # bias
).astype(bool)


# =========================
# 4. MORPHOLOGICAL CLEANUP
# =========================
binary = remove_small_objects(binary, min_size=MIN_COMPONENT_SIZE)
binary = closing(binary, disk(2))


# =========================
# 5. SKELETONIZATION
# =========================
skeleton = skeletonize(binary)
skeleton = remove_small_objects(skeleton, min_size=SKELETON_MIN_SIZE)


# =========================
# 6. SKELETON → GRAPH
# =========================
H, W = skeleton.shape
G = nx.Graph()

def neighbors(y, x):
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx_ = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx_ < W:
                if skeleton[ny, nx_]:
                    yield (ny, nx_)

for y in range(H):
    for x in range(W):
        if skeleton[y, x]:
            G.add_node((y, x))
            for nb in neighbors(y, x):
                G.add_edge((y, x), nb)

endpoints = [n for n in G.nodes if G.degree[n] == 1]
junctions = [n for n in G.nodes if G.degree[n] >= 3]


# =========================
# 7. VISUALIZATION
# =========================
plt.figure(figsize=(22, 6))

# ---- original
plt.subplot(1, 4, 1)
plt.title("Original DSA")
plt.imshow(img, cmap="gray")
plt.axis("off")

# ---- vessel mask
plt.subplot(1, 4, 2)
plt.title("Binary Vessel Mask")
plt.imshow(binary, cmap="gray")
plt.axis("off")

# ---- skeleton
plt.subplot(1, 4, 3)
plt.title("Skeleton")
plt.imshow(img, cmap="gray")
ys, xs = np.where(skeleton)
plt.scatter(xs, ys, s=1, c="cyan")
plt.axis("off")

# ---- graph
plt.subplot(1, 4, 4)
plt.title("Graph (Edges + Nodes)")
plt.imshow(img, cmap="gray")

for (y1, x1), (y2, x2) in G.edges:
    plt.plot([x1, x2], [y1, y2], c="yellow", linewidth=0.3)

if junctions:
    jy, jx = zip(*junctions)
    plt.scatter(jx, jy, c="red", s=12, label="Junctions")

if endpoints:
    ey, ex = zip(*endpoints)
    plt.scatter(ex, ey, c="lime", s=12, label="Endpoints")

plt.legend(loc="lower right")
plt.axis("off")

plt.tight_layout()
plt.show()


# =========================
# 8. PRINT STATS
# =========================
print("Graph stats:")
print(" Nodes:", len(G.nodes))
print(" Edges:", len(G.edges))
print(" Endpoints:", len(endpoints))
print(" Junctions:", len(junctions))
