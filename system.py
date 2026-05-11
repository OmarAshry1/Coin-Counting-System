import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Overlap / containment filters
# ──────────────────────────────────────────────────────────────────────────────
 
def circle_iou(x1, y1, r1, x2, y2, r2):
    """Intersection-over-union for two circles (approximate via area formula)."""
    d = float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):                       # one fully inside the other
        return min(r1, r2) ** 2 / max(r1, r2) ** 2
    a1  = r1 ** 2 * np.arccos(np.clip((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1), -1, 1))
    a2  = r2 ** 2 * np.arccos(np.clip((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2), -1, 1))
    tri = 0.5 * np.sqrt(max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)))
    intersection = a1 + a2 - tri
    union        = np.pi * (r1 ** 2 + r2 ** 2) - intersection
    return intersection / max(union, 1e-6)
 
 
def iou_nms(scored_circles, iou_threshold=0.15):
    """
    Non-maximum suppression using circle IoU.
    `scored_circles` must be sorted descending by score: [(score, x, y, r), …].
    Returns [(x, y, r), …].
    """
    selected = []
    for _score, x, y, r in scored_circles:
        if any(circle_iou(x, y, r, sx, sy, sr) > iou_threshold
               for sx, sy, sr in selected):
            continue
        selected.append((x, y, r))
    return selected
 
 
def remove_contained_circles(circles):
    """Remove circles that are largely contained inside a larger one."""
    filtered = []
    for circle in sorted(circles, key=lambda c: c[2], reverse=True):
        x, y, r = circle
        contained = any(
            np.sqrt((x - fx) ** 2 + (y - fy) ** 2) < 0.95 * fr and r > 0.75 * fr
            for fx, fy, fr in filtered
        )
        if not contained:
            filtered.append(circle)
    return filtered
 
 
def remove_nested_dense_circles(circles):
    filtered = []
    for circle in sorted(circles, key=lambda c: c[2], reverse=True):
        x, y, r = circle
        nested = False
        for fx, fy, fr in filtered:
            dist      = np.sqrt((x - fx) ** 2 + (y - fy) ** 2)
            same_coin = dist < 0.95 * fr and r > 0.75 * fr
            inner_art = dist < 0.78 * fr and r < 0.50 * fr
            if same_coin or inner_art:
                nested = True
                break
        if not nested:
            filtered.append((int(x), int(y), int(r)))
    return filtered
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Pre-processing
# ──────────────────────────────────────────────────────────────────────────────
 
def normalize_illumination(gray):
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=35, sigmaY=35)
    normalized = cv2.divide(gray, background, scale=180)
    return cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
 
 
def preprocess_image(image, normalize=False):
    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if normalize:
        gray = normalize_illumination(gray)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur     = cv2.medianBlur(enhanced, 5)
    edges    = cv2.Canny(blur, 50, 130)
    bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)   # used by GRADIENT_ALT
    return gray, blur, bilateral, edges
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Radius range  (BUG 1 + BUG 6 fix)
# ──────────────────────────────────────────────────────────────────────────────
 
def estimate_radius_range(image_shape, min_radius=None, max_radius=None):
    height, width = image_shape[:2]
    shortest_side = min(height, width)
 
    if min_radius is None:
        # BUG 6 FIX: was 0.035 → now 0.12 to skip internal coin artwork
        min_radius = max(12, int(shortest_side * 0.12))
 
    if max_radius is None:
        # BUG 1 FIX: was 0.22 → now 0.35 to capture large coins
        max_radius = max(min_radius + 5, int(shortest_side * 0.35))
 
    return min_radius, max_radius
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Hough circle candidates  (BUG 2 + BUG 3 fix)
# ──────────────────────────────────────────────────────────────────────────────
 
def find_circle_candidates(blur, bilateral, min_radius, max_radius, dense=False):
    """
    Collect circle candidates from both GRADIENT and GRADIENT_ALT detectors,
    across multiple param2 and min_dist values, then deduplicate by proximity.
 
    BUG 2 FIX: we no longer stop at the first non-empty param2 result.
    BUG 3 FIX: multiple min_dist passes so nearby coins survive Hough's internal
               suppression.
    """
    if dense:
        return _find_dense_candidates(blur, min_radius, max_radius)
 
    all_circles = []
 
    # --- HOUGH_GRADIENT: multiple param2 × multiple min_dist ---
    param2_values  = (78, 68, 58, 50, 42, 35)
    min_dist_values = (15, 25, 40)          # BUG 3: was single formula-based value
 
    for p2 in param2_values:
        for min_dist in min_dist_values:
            circles = cv2.HoughCircles(
                blur,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=min_dist,
                param1=120,
                param2=p2,
                minRadius=min_radius,
                maxRadius=max_radius,
            )
            if circles is not None:
                all_circles.extend(np.round(circles[0, :]).astype(int).tolist())
 
    # --- HOUGH_GRADIENT_ALT: more accurate centers for round coins ---
    for p2 in (0.60, 0.65, 0.70, 0.75):
        circles = cv2.HoughCircles(
            bilateral,
            cv2.HOUGH_GRADIENT_ALT,
            dp=1.5,
            minDist=20,
            param1=300,
            param2=p2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is not None:
            all_circles.extend(np.round(circles[0, :]).astype(int).tolist())
 
    if not all_circles:
        return np.empty((0, 3), dtype=int)
 
    # Deduplicate by proximity (keep first occurrence = highest-scoring param2)
    unique = []
    for cx, cy, r in all_circles:
        if not any(
            np.sqrt((cx - ux) ** 2 + (cy - uy) ** 2) < 0.45 * max(r, ur)
            for ux, uy, ur in unique
        ):
            unique.append((cx, cy, r))
 
    return np.array(unique, dtype=int)
 
 
def _find_dense_candidates(blur, min_radius, max_radius):
    """Candidate finder for densely packed coin piles."""
    min_dist  = max(int(max_radius * 0.55), int(min_radius * 1.8), 20)
    candidates = []
 
    for param2 in (74, 66, 58, 50, 42):
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dist,
            param1=130, param2=param2,
            minRadius=min_radius, maxRadius=max_radius,
        )
        if circles is not None:
            candidates.extend(np.round(circles[0, :]).astype(int))
 
    if not candidates:
        return np.empty((0, 3), dtype=int)
 
    # Deduplicate near-identicals
    unique = []
    for circle in candidates:
        x, y, r  = circle
        duplicate = any(
            np.sqrt((x - ux) ** 2 + (y - uy) ** 2) <= 8 and abs(r - ur) <= 8
            for ux, uy, ur in unique
        )
        if not duplicate:
            unique.append((x, y, r))
    return np.array(unique, dtype=int)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Edge / gradient quality scoring
# ──────────────────────────────────────────────────────────────────────────────
 
def circle_edge_density(edges, x, y, r):
    """Fraction of pixels in the coin's rim annulus that are edge pixels."""
    outer = np.zeros(edges.shape, dtype="uint8")
    inner = np.zeros(edges.shape, dtype="uint8")
    cv2.circle(outer, (x, y), r + 2, 255, -1)
    cv2.circle(inner, (x, y), max(1, r - 3), 255, -1)
    ring      = cv2.subtract(outer, inner)
    edge_pix  = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=ring))
    ring_pix  = max(1, cv2.countNonZero(ring))
    return edge_pix / ring_pix
 
 
def circle_edge_support(edges, gradient_x, gradient_y, x, y, r, bins=36):
    """Returns (angular_coverage, radial_alignment) of edge pixels near the rim."""
    edge_y, edge_x = np.where(edges > 0)
    dx = edge_x - x
    dy = edge_y - y
    distances  = np.sqrt(dx * dx + dy * dy)
    ring_mask  = (distances >= r - 3) & (distances <= r + 3) & (distances > 1)
 
    if not np.any(ring_mask):
        return 0.0, 0.0
 
    angles    = (np.arctan2(dy[ring_mask], dx[ring_mask]) + np.pi) / (2 * np.pi)
    histogram = np.histogram(angles, bins=bins, range=(0, 1))[0]
    coverage  = np.count_nonzero(histogram) / bins
 
    xs = edge_x[ring_mask];  ys = edge_y[ring_mask]
    rdx = dx[ring_mask] / distances[ring_mask]
    rdy = dy[ring_mask] / distances[ring_mask]
    gmag = np.sqrt(gradient_x[ys, xs] ** 2 + gradient_y[ys, xs] ** 2) + 1e-6
    gux  = gradient_x[ys, xs] / gmag
    guy  = gradient_y[ys, xs] / gmag
    radial_alignment = float(np.mean(np.abs(rdx * gux + rdy * guy)))
 
    return float(coverage), radial_alignment
 
 
def score_circle_candidate(edges, gradient_x, gradient_y, x, y, r):
    density                 = circle_edge_density(edges, x, y, r)
    coverage, radial_align  = circle_edge_support(edges, gradient_x, gradient_y, x, y, r)
    large_weak_penalty      = 0.25 if r > 130 and density < 0.19 else 0
    return (1.2 * density) + (0.7 * coverage) + (0.9 * radial_align) - large_weak_penalty
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Coin validation  (BUG 5 fix)
# ──────────────────────────────────────────────────────────────────────────────
 
def is_valid_coin(gray, edges, x, y, r, dense=False, median_radius=None):
    height, width = gray.shape
    # BUG 5 FIX: was r*0.08, now r*0.15 → allows partially clipped border coins
    margin = max(2, int(r * 0.15))
 
    if (x - r < -margin or y - r < -margin
            or x + r >= width  + margin
            or y + r >= height + margin):
        return False
 
    coin_mask = np.zeros(gray.shape, dtype="uint8")
    cv2.circle(coin_mask, (x, y), int(r * 0.85), 255, -1)
    _mean, std_intensity = cv2.meanStdDev(gray, mask=coin_mask)
    edge_density         = circle_edge_density(edges, x, y, r)
 
    if dense and median_radius and r >= 0.85 * median_radius and edge_density < 0.14:
        return False
 
    return std_intensity[0][0] > 10 and edge_density > 0.08
 
 
def select_dense_candidates(gray, edges, candidates):
    gradient_x = cv2.Sobel(gray.astype("float32"), cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray.astype("float32"), cv2.CV_32F, 0, 1, ksize=3)
    scored = []
 
    for x, y, r in candidates:
        if not is_valid_coin(gray, edges, x, y, r):
            continue
        score = score_circle_candidate(edges, gradient_x, gradient_y, x, y, r)
        if score >= 1.35:
            scored.append((score, x, y, r))
 
    selected = []
    for score, x, y, r in sorted(scored, reverse=True):
        keep = not any(
            np.sqrt((x - sx) ** 2 + (y - sy) ** 2) <= 0.65 * min(r, sr)
            and abs(r - sr) <= 0.35 * max(r, sr)
            for sx, sy, sr in selected
        )
        if keep:
            selected.append((x, y, r))
 
    return selected
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Radius refinement
# ──────────────────────────────────────────────────────────────────────────────
 
def refine_small_coin_radii(edges, circles):
    if not circles:
        return circles
 
    median_radius = np.median([r for _, _, r in circles])
    refined       = []
 
    for x, y, r in circles:
        if r > 0.75 * median_radius:
            refined.append((x, y, r))
            continue
        best_r, best_score = r, -1
        for test_r in range(max(8, int(r * 0.85)), max(int(r * 0.85) + 1, int(r * 1.10)) + 1):
            s = circle_edge_density(edges, x, y, test_r)
            if s > best_score:
                best_score = s
                best_r     = test_r
        refined.append((x, y, best_r))
 
    return refined
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Color / material classification
# ──────────────────────────────────────────────────────────────────────────────
 
def extract_color_features(image, circles):
    hsv   = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    feats = []
    for (x, y, r) in circles:
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.circle(mask, (x, y), int(r * 0.85), 255, -1)
        h, s, v, _ = cv2.mean(hsv, mask=mask)
        feats.append([h, s, v])
    return np.array(feats)
 
 
def classify_materials(color_feats):
    material_to_label = {"Gold": 0, "Silver": 1, "Bronze": 2}
    color_names       = {0: "Gold", 1: "Silver", 2: "Bronze"}
    labels            = []
 
    for h, s, v in color_feats:
        if   s < 85 and v > 115:
            material = "Silver"
        elif 15 <= h <= 45 and s > 80 and v > 130:
            material = "Gold"
        else:
            material = "Bronze"
        labels.append(material_to_label[material])
 
    return np.array(labels), color_names
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Drawing  (BUG 7 fix: consistent radius scaling + per-material colours)
# ──────────────────────────────────────────────────────────────────────────────
 
MATERIAL_COLORS_BGR = {
    "Gold":   (0,  185, 220),   # golden-orange
    "Silver": (210, 210, 210),  # light grey
    "Bronze": (30,  105, 180),  # warm brown
}
 
 
def sort_circles(circles):
    return sorted(circles, key=lambda c: (c[1], c[0]))
 
 
def draw_coin_results(image, coin_results, total, dense=False):
    h, w       = image.shape[:2]
    panel_w    = 280 if total > 25 else 250
    canvas     = np.full((h, w + panel_w, 3), 255, dtype=np.uint8)
    canvas[:, :w] = image
 
    cv2.putText(canvas, f"Coins: {total}", (w + 16, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
 
    thick      = 1 if total > 25 else 2
    brad       = 9 if total > 25 else 12
    bfont      = 0.35 if total > 25 else 0.45
    row_start  = 52 if total > 25 else 66
    row_step   = 18 if total > 25 else 24
    row_font   = 0.4 if total > 25 else 0.5
 
    for coin in coin_results:
        idx, (x, y, r), material, size = (
            coin["id"], coin["circle"], coin["material"], coin["size"]
        )
        color = MATERIAL_COLORS_BGR.get(material, (0, 160, 255))
 
        # BUG 7 FIX: clip circle to canvas bounds before drawing
        x_c, y_c = int(np.clip(x, 0, w + panel_w - 1)), int(np.clip(y, 0, h - 1))
 
        if not dense:
            cv2.circle(canvas, (x_c, y_c), r, color, thick)
 
        # Badge (number label at coin center)
        cv2.circle(canvas, (x_c, y_c), brad, color, -1)
        cv2.circle(canvas, (x_c, y_c), brad, (255, 255, 255), 1)
        cv2.putText(canvas, str(idx),
                    (x_c - 5 if idx < 10 else x_c - 8, y_c + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, bfont, (20, 20, 20),
                    1 if total > 25 else 2)
 
        row_y = row_start + (idx - 1) * row_step
        if row_y < h - 12:
            cv2.putText(canvas, f"{idx}. {material} - {size}",
                        (w + 18, row_y), cv2.FONT_HERSHEY_SIMPLEX,
                        row_font, (30, 30, 30), 1)
 
    return canvas
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────
 
def detect_coins(
    image_path,
    output_path=None,
    show=True,
    min_radius=None,
    max_radius=None,
    debug=False,
    mode="auto",
):
    image_path = Path(image_path)
    if output_path is None:
        output_path = Path("output") / f"{image_path.stem}_detected.jpg"
    else:
        output_path = Path(output_path)
 
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path.resolve()}")
 
    min_radius, max_radius = estimate_radius_range(image.shape, min_radius, max_radius)
 
    if mode == "auto":
        dense = min(image.shape[:2]) >= 650
    else:
        dense = mode == "dense"
 
    gray, blur, bilateral, edges = preprocess_image(
        image, normalize=not dense and min(image.shape[:2]) >= 500
    )
 
    circles = find_circle_candidates(blur, bilateral, min_radius, max_radius, dense=dense)
 
    if debug:
        print(f"Image      : {image.shape[1]}×{image.shape[0]}")
        print(f"Radius range: {min_radius}–{max_radius} px")
        print(f"Mode       : {'dense' if dense else 'balanced'}")
        print(f"Hough candidates: {len(circles)}")
 
    if len(circles) == 0:
        print("No circles detected.")
        return None
 
    # ── Compute gradients for scoring ────────────────────────────────────────
    gradient_x = cv2.Sobel(gray.astype("float32"), cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray.astype("float32"), cv2.CV_32F, 0, 1, ksize=3)
 
    if dense:
        valid_circles = select_dense_candidates(gray, edges, circles)
        valid_circles = [(x, y, int(r * 1.03)) for x, y, r in valid_circles]
        valid_circles = remove_nested_dense_circles(valid_circles)
    else:
        # Score every candidate, filter invalid, then IoU-NMS (BUG 4 fix)
        scored = [
            (score_circle_candidate(edges, gradient_x, gradient_y, x, y, r), x, y, r)
            for x, y, r in circles
        ]
        scored = [
            (s, x, y, r) for s, x, y, r in scored
            if s >= 1.25 and is_valid_coin(gray, edges, x, y, r)
        ]
        scored.sort(reverse=True)
 
        valid_circles = iou_nms(scored, iou_threshold=0.15)   # BUG 4 FIX
        valid_circles = remove_contained_circles(valid_circles)
        # BUG 7 FIX: scale radius AFTER selection, not before scoring
        valid_circles = [(x, y, int(r * 1.03)) for x, y, r in valid_circles]
 
    valid_circles = refine_small_coin_radii(edges, valid_circles)
 
    if not valid_circles:
        print("No valid coins found.")
        return None
 
    valid_circles = sort_circles(valid_circles)
    radii         = np.array([r for _, _, r in valid_circles])
    median_r      = np.median(radii)
 
    color_feats         = extract_color_features(image, valid_circles)
    labels, color_names = classify_materials(color_feats)
 
    total           = len(valid_circles)
    material_counts = {"Gold": 0, "Silver": 0, "Bronze": 0}
    size_counts     = {"Big": 0, "Small": 0}
    combo_counts    = {
        "Gold_Big": 0, "Gold_Small": 0,
        "Silver_Big": 0, "Silver_Small": 0,
        "Bronze_Big": 0, "Bronze_Small": 0,
    }
    coin_results = []
 
    for i, (x, y, r) in enumerate(valid_circles):
        material = color_names[labels[i]]
        size     = "Big" if r >= median_r else "Small"
        material_counts[material]          += 1
        size_counts[size]                  += 1
        combo_counts[f"{material}_{size}"] += 1
        coin_results.append({"id": i + 1, "circle": (x, y, r),
                              "material": material, "size": size})
 
    print(f"\nTotal coins: {total}")
    print("\nBy material:")
    print(material_counts)
    print("\nBy size:")
    print(size_counts)
    print("\nDetailed:")
    print(combo_counts)
 
    output_img = draw_coin_results(image, coin_results, total, dense=dense)
 
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output_img)
    print(f"\nSaved annotated image → {output_path}")
 
    if show:
        plt.figure(figsize=(13, 6))
        plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"Coins detected: {total}")
        plt.tight_layout()
        preview = str(output_path).replace(".jpg", "_preview.png")
        plt.savefig(preview, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Preview saved  → {preview}")
 
    return {
        "total"          : total,
        "material_counts": material_counts,
        "size_counts"    : size_counts,
        "combo_counts"   : combo_counts,
        "output_path"    : str(output_path),
        "radius_range"   : (min_radius, max_radius),
    }
 
 
# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
 
def parse_args():
    p = argparse.ArgumentParser(description="Detect and classify coins in a local image.")
    p.add_argument("image",       nargs="?", default="images/1.jpg")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--no-show",   action="store_true")
    p.add_argument("--min-radius", type=int, default=None)
    p.add_argument("--max-radius", type=int, default=None)
    p.add_argument("--debug",     action="store_true")
    p.add_argument("--mode",      choices=("auto", "balanced", "dense"), default="auto")
    return p.parse_args()
 
 
if __name__ == "__main__":
    args = parse_args()
    detect_coins(
        args.image,
        args.output,
        show=not args.no_show,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        debug=args.debug,
        mode=args.mode,
    )