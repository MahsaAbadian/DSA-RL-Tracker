#!/usr/bin/env python3
"""
Enhanced DSA Skeleton Extraction
Improved vessel segmentation and skeletonization for DSA images
"""

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

from skimage.filters import frangi, threshold_multiotsu, threshold_otsu
from skimage.morphology import (
    skeletonize, remove_small_objects, closing, opening, 
    disk, binary_dilation, binary_erosion
)
from skimage.measure import label, regionprops
from scipy import ndimage


# =========================
# PREPROCESSING
# =========================
def preprocess_dsa(img, clahe_clip=2.0, invert_if_needed=True):
    """
    Preprocess DSA image with CLAHE and inversion.
    
    Args:
        img: Input grayscale image (0-1 float or 0-255 uint8)
        clahe_clip: CLAHE clip limit
        invert_if_needed: Auto-invert if light background
    
    Returns:
        Preprocessed image (0-1 float, dark background)
    """
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    
    # CLAHE for contrast enhancement
    img_uint8 = (img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_uint8).astype(np.float32) / 255.0
    
    # Auto-invert if light background (vessels should be bright on dark)
    if invert_if_needed and np.median(img_enhanced) > 0.5:
        print("  → Detected light background, inverting...")
        img_enhanced = 1.0 - img_enhanced
    
    return img_enhanced


# =========================
# VESSEL ENHANCEMENT
# =========================
def multi_scale_frangi(img, scale_ranges=None, return_best=True):
    """
    Apply Frangi filter at multiple scale ranges.
    
    Args:
        img: Input image (0-1 float)
        scale_ranges: List of (min, max) scale tuples
        return_best: If True, return max response; else return list
    
    Returns:
        Vesselness map(s)
    """
    if scale_ranges is None:
        # Default scales for DSA: thin to thick vessels
        scale_ranges = [(1, 3), (2, 5), (4, 8)]
    
    responses = []
    for scale_min, scale_max in scale_ranges:
        print(f"  → Frangi filter: scales {scale_min}-{scale_max}")
        vesselness = frangi(
            img,
            sigmas=range(scale_min, scale_max + 1),
            alpha=0.5,
            beta=0.5,
            gamma=15,
            black_ridges=False
        )
        responses.append(vesselness)
    
    if return_best:
        # Take maximum response across scales
        combined = np.maximum.reduce(responses)
        return (combined - combined.min()) / (combined.max() + 1e-8)
    else:
        return responses


# =========================
# THRESHOLDING
# =========================
def threshold_vessels(vesselness, method='multi_otsu', **kwargs):
    """
    Threshold vesselness map to binary mask.
    
    Args:
        vesselness: Vesselness map (0-1 float)
        method: 'otsu', 'multi_otsu', 'adaptive', 'hysteresis'
        **kwargs: Method-specific parameters
    
    Returns:
        Binary vessel mask
    """
    v_uint8 = (vesselness * 255).astype(np.uint8)
    
    if method == 'otsu':
        thresh = threshold_otsu(vesselness)
        binary = vesselness > thresh
        print(f"  → Otsu threshold: {thresh:.3f}")
    
    elif method == 'multi_otsu':
        # Use highest class from multi-Otsu
        n_classes = kwargs.get('n_classes', 3)
        thresholds = threshold_multiotsu(vesselness, classes=n_classes)
        binary = vesselness > thresholds[-2]  # Use second-highest threshold
        print(f"  → Multi-Otsu thresholds: {[f'{t:.3f}' for t in thresholds]}")
    
    elif method == 'adaptive':
        block_size = kwargs.get('block_size', 41)
        bias = kwargs.get('bias', -5)
        binary = cv2.adaptiveThreshold(
            v_uint8, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size, bias
        ).astype(bool)
        print(f"  → Adaptive threshold: block={block_size}, bias={bias}")
    
    elif method == 'hysteresis':
        # Two-threshold approach (like Canny edge detection)
        low = kwargs.get('low_thresh', 0.1)
        high = kwargs.get('high_thresh', 0.3)
        high_mask = vesselness > high
        low_mask = vesselness > low
        # Grow from high seeds
        labeled = label(low_mask)
        high_labels = np.unique(labeled[high_mask])
        binary = np.isin(labeled, high_labels[high_labels > 0])
        print(f"  → Hysteresis: low={low:.3f}, high={high:.3f}")
    
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    
    return binary


# =========================
# MORPHOLOGICAL CLEANUP
# =========================
def morphological_cleanup(binary, min_size=200, closing_radius=2, opening_radius=1):
    """
    Clean up binary mask with morphological operations.
    
    Args:
        binary: Binary vessel mask
        min_size: Minimum component size to keep
        closing_radius: Radius for closing operation
        opening_radius: Radius for opening operation
    
    Returns:
        Cleaned binary mask
    """
    print(f"  → Removing small objects (min size: {min_size})")
    cleaned = remove_small_objects(binary, min_size=min_size)
    
    if closing_radius > 0:
        print(f"  → Morphological closing (radius: {closing_radius})")
        cleaned = closing(cleaned, disk(closing_radius))
    
    if opening_radius > 0:
        print(f"  → Morphological opening (radius: {opening_radius})")
        cleaned = opening(cleaned, disk(opening_radius))
    
    return cleaned


# =========================
# SKELETONIZATION
# =========================
def extract_skeleton(binary, min_skeleton_size=40):
    """
    Extract skeleton from binary mask.
    
    Args:
        binary: Binary vessel mask
        min_skeleton_size: Minimum skeleton component size
    
    Returns:
        Skeleton (binary)
    """
    print("  → Skeletonizing...")
    skeleton = skeletonize(binary)
    
    print(f"  → Removing small skeleton fragments (min size: {min_skeleton_size})")
    skeleton = remove_small_objects(skeleton, min_size=min_skeleton_size)
    
    return skeleton


# =========================
# GRAPH CONSTRUCTION
# =========================
def skeleton_to_graph(skeleton):
    """
    Convert skeleton to NetworkX graph.
    
    Args:
        skeleton: Binary skeleton image
    
    Returns:
        G: NetworkX graph
        endpoints: List of endpoint nodes
        junctions: List of junction nodes
    """
    print("  → Building graph from skeleton...")
    H, W = skeleton.shape
    G = nx.Graph()
    
    def neighbors(y, x):
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx_ < W and skeleton[ny, nx_]:
                    yield (ny, nx_)
    
    # Build graph
    for y in range(H):
        for x in range(W):
            if skeleton[y, x]:
                G.add_node((y, x))
                for nb in neighbors(y, x):
                    G.add_edge((y, x), nb)
    
    # Identify special nodes
    endpoints = [n for n in G.nodes if G.degree[n] == 1]
    junctions = [n for n in G.nodes if G.degree[n] >= 3]
    
    print(f"     Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
    print(f"     Endpoints: {len(endpoints)}, Junctions: {len(junctions)}")
    
    return G, endpoints, junctions


# =========================
# GRAPH CLEANUP
# =========================
def prune_short_branches(G, endpoints, min_length=10):
    """
    Remove short terminal branches (spurs) from graph.
    
    Args:
        G: NetworkX graph
        endpoints: List of endpoint nodes
        min_length: Minimum branch length to keep
    
    Returns:
        Pruned graph, new endpoints
    """
    G = G.copy()
    pruned_count = 0
    
    for endpoint in endpoints:
        if endpoint not in G:
            continue
        
        # Trace path from endpoint until junction or another endpoint
        path = [endpoint]
        current = endpoint
        visited = {endpoint}
        
        while True:
            neighbors = [n for n in G.neighbors(current) if n not in visited]
            if len(neighbors) == 0:
                break
            
            # Check if we reached a junction
            if G.degree[current] > 2:
                break
            
            next_node = neighbors[0]
            path.append(next_node)
            visited.add(next_node)
            current = next_node
        
        # If path is short, remove it
        if len(path) < min_length and G.degree[path[-1]] > 2:
            for node in path[:-1]:  # Keep the junction
                G.remove_node(node)
            pruned_count += 1
    
    # Update endpoints
    new_endpoints = [n for n in G.nodes if G.degree[n] == 1]
    
    if pruned_count > 0:
        print(f"  → Pruned {pruned_count} short branches")
    
    return G, new_endpoints


def connect_nearby_endpoints(G, endpoints, max_distance=15):
    """
    Connect nearby endpoints to reduce fragmentation.
    
    Args:
        G: NetworkX graph
        endpoints: List of endpoint nodes
        max_distance: Maximum distance to connect
    
    Returns:
        Graph with connected endpoints
    """
    G = G.copy()
    connected_count = 0
    
    endpoints_array = np.array(endpoints)
    if len(endpoints_array) < 2:
        return G, connected_count
    
    # Compute pairwise distances
    for i, ep1 in enumerate(endpoints):
        if ep1 not in G:
            continue
        
        for ep2 in endpoints[i+1:]:
            if ep2 not in G:
                continue
            
            dist = np.linalg.norm(np.array(ep1) - np.array(ep2))
            
            if dist <= max_distance:
                # Add edge
                G.add_edge(ep1, ep2)
                connected_count += 1
    
    if connected_count > 0:
        print(f"  → Connected {connected_count} nearby endpoint pairs")
    
    return G, connected_count


# =========================
# VISUALIZATION
# =========================
def visualize_pipeline(img, vesselness, binary, skeleton, G, endpoints, junctions, 
                       save_path=None, show=True):
    """
    Visualize complete pipeline.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Processing steps
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('1. Preprocessed Image', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(vesselness, cmap='hot')
    axes[0, 1].set_title('2. Vessel Enhancement (Frangi)', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(binary, cmap='gray')
    axes[0, 2].set_title('3. Binary Vessel Mask', fontsize=12)
    axes[0, 2].axis('off')
    
    # Row 2: Skeleton and graph
    axes[1, 0].imshow(img, cmap='gray', alpha=0.7)
    ys, xs = np.where(skeleton)
    axes[1, 0].scatter(xs, ys, s=1, c='cyan', alpha=0.8)
    axes[1, 0].set_title('4. Skeleton Overlay', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(skeleton, cmap='gray')
    axes[1, 1].set_title('5. Skeleton', fontsize=12)
    axes[1, 1].axis('off')
    
    # Graph visualization
    axes[1, 2].imshow(img, cmap='gray', alpha=0.7)
    
    # Draw edges
    for (y1, x1), (y2, x2) in G.edges:
        axes[1, 2].plot([x1, x2], [y1, y2], c='yellow', linewidth=0.5, alpha=0.6)
    
    # Draw junctions
    if junctions:
        jy, jx = zip(*junctions)
        axes[1, 2].scatter(jx, jy, c='red', s=20, label=f'Junctions ({len(junctions)})', 
                          edgecolors='white', linewidths=0.5, zorder=3)
    
    # Draw endpoints
    if endpoints:
        ey, ex = zip(*endpoints)
        axes[1, 2].scatter(ex, ey, c='lime', s=20, label=f'Endpoints ({len(endpoints)})',
                          edgecolors='white', linewidths=0.5, zorder=3)
    
    axes[1, 2].set_title('6. Graph Structure', fontsize=12)
    axes[1, 2].legend(loc='upper right', fontsize=9)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved visualization: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# =========================
# MAIN PIPELINE
# =========================
def extract_dsa_skeleton(
    image_path,
    clahe_clip=2.0,
    scale_ranges=None,
    threshold_method='multi_otsu',
    min_component_size=200,
    closing_radius=2,
    opening_radius=1,
    min_skeleton_size=40,
    prune_length=10,
    connect_distance=15,
    save_dir=None,
    show=True
):
    """
    Complete DSA skeleton extraction pipeline.
    
    Args:
        image_path: Path to DSA image
        clahe_clip: CLAHE clip limit (0-5, default 2.0)
        scale_ranges: List of (min, max) scale tuples for Frangi
        threshold_method: 'otsu', 'multi_otsu', 'adaptive', 'hysteresis'
        min_component_size: Minimum vessel component size (pixels)
        closing_radius: Morphological closing disk radius
        opening_radius: Morphological opening disk radius
        min_skeleton_size: Minimum skeleton fragment size (pixels)
        prune_length: Minimum branch length (pixels)
        connect_distance: Max distance to connect endpoints (pixels)
        save_dir: Directory to save results (None = no save)
        show: Whether to display results
    
    Returns:
        dict with all intermediate and final results
    """
    print("="*60)
    print("DSA SKELETON EXTRACTION")
    print("="*60)
    
    # Load image
    print(f"\n1. Loading: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Preprocess
    print("\n2. Preprocessing...")
    img_processed = preprocess_dsa(img, clahe_clip=clahe_clip)
    
    # Vessel enhancement
    print("\n3. Vessel enhancement (Frangi filter)...")
    vesselness = multi_scale_frangi(img_processed, scale_ranges=scale_ranges)
    
    # Threshold
    print("\n4. Thresholding...")
    binary = threshold_vessels(vesselness, method=threshold_method)
    
    # Morphological cleanup
    print("\n5. Morphological cleanup...")
    binary_clean = morphological_cleanup(
        binary,
        min_size=min_component_size,
        closing_radius=closing_radius,
        opening_radius=opening_radius
    )
    
    # Skeletonize
    print("\n6. Skeletonization...")
    skeleton = extract_skeleton(binary_clean, min_skeleton_size=min_skeleton_size)
    
    # Build graph
    print("\n7. Graph construction...")
    G, endpoints, junctions = skeleton_to_graph(skeleton)
    
    # Graph cleanup
    print("\n8. Graph cleanup...")
    if prune_length > 0:
        G, endpoints = prune_short_branches(G, endpoints, min_length=prune_length)
        junctions = [n for n in G.nodes if G.degree[n] >= 3]
    
    if connect_distance > 0:
        G, _ = connect_nearby_endpoints(G, endpoints, max_distance=connect_distance)
        endpoints = [n for n in G.nodes if G.degree[n] == 1]
        junctions = [n for n in G.nodes if G.degree[n] >= 3]
    
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    print(f"Graph nodes:     {len(G.nodes)}")
    print(f"Graph edges:     {len(G.edges)}")
    print(f"Endpoints:       {len(endpoints)}")
    print(f"Junctions:       {len(junctions)}")
    print(f"Connected comp:  {nx.number_connected_components(G)}")
    print("="*60)
    
    # Visualization
    print("\n9. Visualization...")
    save_path = None
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        img_name = Path(image_path).stem
        save_path = save_dir / f"{img_name}_skeleton_pipeline.png"
    
    visualize_pipeline(
        img_processed, vesselness, binary_clean, skeleton, 
        G, endpoints, junctions,
        save_path=save_path, show=show
    )
    
    # Return all results
    return {
        'image': img_processed,
        'vesselness': vesselness,
        'binary': binary_clean,
        'skeleton': skeleton,
        'graph': G,
        'endpoints': endpoints,
        'junctions': junctions
    }


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Enhanced DSA Skeleton Extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('image_path', nargs='?', default="/Users/geshvad/Desktop/test2.png", 
                       type=str, help='Path to DSA image (default: image_s40_i5.png)')
    parser.add_argument('--clahe_clip', type=float, default=2.0, 
                       help='CLAHE clip limit')
    parser.add_argument('--threshold', type=str, default='multi_otsu',
                       choices=['otsu', 'multi_otsu', 'adaptive', 'hysteresis'],
                       help='Thresholding method')
    parser.add_argument('--min_size', type=int, default=200,
                       help='Minimum vessel component size (pixels)')
    parser.add_argument('--closing', type=int, default=2,
                       help='Morphological closing radius')
    parser.add_argument('--opening', type=int, default=1,
                       help='Morphological opening radius')
    parser.add_argument('--min_skeleton', type=int, default=40,
                       help='Minimum skeleton fragment size (pixels)')
    parser.add_argument('--prune', type=int, default=10,
                       help='Minimum branch length (0=no pruning)')
    parser.add_argument('--connect', type=int, default=15,
                       help='Max distance to connect endpoints (0=no connection)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save results')
    parser.add_argument('--no_show', action='store_true',
                       help='Do not display visualization')
    
    args = parser.parse_args()
    
    # Run pipeline
    results = extract_dsa_skeleton(
        args.image_path,
        clahe_clip=args.clahe_clip,
        threshold_method=args.threshold,
        min_component_size=args.min_size,
        closing_radius=args.closing,
        opening_radius=args.opening,
        min_skeleton_size=args.min_skeleton,
        prune_length=args.prune,
        connect_distance=args.connect,
        save_dir=args.save_dir,
        show=not args.no_show
    )
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()

