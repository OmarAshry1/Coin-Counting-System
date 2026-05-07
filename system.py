import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


# Remove duplicate circles
def remove_overlapping_circles(circles, overlap_factor=0.55):
    if len(circles) == 0:
        return circles

    circles = sorted(circles, key=lambda x: x[2], reverse=True)
    filtered = []

    for (x, y, r) in circles:
        keep = True
        for (fx, fy, fr) in filtered:
            dist = np.sqrt((x - fx)**2 + (y - fy)**2)

            if dist < overlap_factor * max(r, fr):
                keep = False
                break

        if keep:
            filtered.append((x, y, r))

    return np.array(filtered)


def remove_near_identical_circles(circles, center_tolerance=8, radius_tolerance=8):
    unique = []

    for circle in circles:
        x, y, r = circle
        duplicate = False

        for ux, uy, ur in unique:
            dist = np.sqrt((x - ux) ** 2 + (y - uy) ** 2)
            if dist <= center_tolerance and abs(r - ur) <= radius_tolerance:
                duplicate = True
                break

        if not duplicate:
            unique.append((x, y, r))

    return np.array(unique, dtype=int)



def normalize_illumination(gray):
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=35, sigmaY=35)
    normalized = cv2.divide(gray, background, scale=180)
    return cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")


def preprocess_image(image, normalize=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if normalize:
        gray = normalize_illumination(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.medianBlur(enhanced, 5)
    edges = cv2.Canny(blur, 70, 160)
    return gray, blur, edges


def estimate_radius_range(image_shape, min_radius=None, max_radius=None):
    height, width = image_shape[:2]
    shortest_side = min(height, width)

    if min_radius is None:
        min_radius = max(8, int(shortest_side * 0.035))
    if max_radius is None:
        max_radius = max(min_radius + 5, int(shortest_side * 0.22))

    return min_radius, max_radius


def find_circle_candidates(blur, min_radius, max_radius, dense=False):
    if dense:
        return find_dense_circle_candidates(blur, min_radius, max_radius)

    if dense:
        min_dist = max(int(max_radius * 0.55), int(min_radius * 1.8), 20)
        hough_thresholds = (58, 50, 42)
        overlap_factor = 0.55
        radius_cutoff = 0.45
    else:
        min_dist = max(int(max_radius * 0.55), int(min_radius * 1.8), 20)
        hough_thresholds = (74, 66, 58, 50, 42)
        overlap_factor = 0.75
        radius_cutoff = 0.55

    # A moderately strict threshold usually finds real coin rims better than
    # very strict settings, which can latch onto internal coin artwork.
    for param2 in hough_thresholds:
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dist,
            param1=130,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        if circles is not None:
            candidates = np.round(circles[0, :]).astype("int")
            candidates = remove_overlapping_circles(np.array(candidates), overlap_factor=overlap_factor)
            if len(candidates) > 0:
                median_radius = np.median(candidates[:, 2])
                candidates = candidates[candidates[:, 2] >= radius_cutoff * median_radius]
            return candidates

    return np.empty((0, 3), dtype=int)


def find_dense_circle_candidates(blur, min_radius, max_radius):
    min_dist = max(int(max_radius * 0.55), int(min_radius * 1.8), 20)
    candidates = []

    for param2 in (74, 66, 58, 50, 42):
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dist,
            param1=130,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        if circles is not None:
            candidates.extend(np.round(circles[0, :]).astype("int"))

    if len(candidates) == 0:
        return np.empty((0, 3), dtype=int)

    return remove_near_identical_circles(candidates)


def circle_edge_density(edges, x, y, r):
    edge_outer = np.zeros(edges.shape, dtype="uint8")
    edge_inner = np.zeros(edges.shape, dtype="uint8")
    cv2.circle(edge_outer, (x, y), r + 2, 255, -1)
    cv2.circle(edge_inner, (x, y), max(1, r - 3), 255, -1)
    ring_mask = cv2.subtract(edge_outer, edge_inner)

    edge_pixels = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=ring_mask))
    expected_ring_pixels = max(1, cv2.countNonZero(ring_mask))
    return edge_pixels / expected_ring_pixels


def circle_edge_support(edges, gradient_x, gradient_y, x, y, r, bins=36):
    edge_y, edge_x = np.where(edges > 0)
    dx = edge_x - x
    dy = edge_y - y
    distances = np.sqrt(dx * dx + dy * dy)
    ring_mask = (distances >= r - 3) & (distances <= r + 3) & (distances > 1)

    if np.count_nonzero(ring_mask) == 0:
        return 0.0, 0.0

    angles = (np.arctan2(dy[ring_mask], dx[ring_mask]) + np.pi) / (2 * np.pi)
    histogram = np.histogram(angles, bins=bins, range=(0, 1))[0]
    coverage = np.count_nonzero(histogram) / bins

    xs = edge_x[ring_mask]
    ys = edge_y[ring_mask]
    radial_x = dx[ring_mask] / distances[ring_mask]
    radial_y = dy[ring_mask] / distances[ring_mask]
    gradient_magnitude = np.sqrt(gradient_x[ys, xs] ** 2 + gradient_y[ys, xs] ** 2) + 1e-6
    gradient_unit_x = gradient_x[ys, xs] / gradient_magnitude
    gradient_unit_y = gradient_y[ys, xs] / gradient_magnitude
    radial_alignment = np.mean(np.abs(radial_x * gradient_unit_x + radial_y * gradient_unit_y))

    return float(coverage), float(radial_alignment)


def score_circle_candidate(edges, gradient_x, gradient_y, x, y, r):
    density = circle_edge_density(edges, x, y, r)
    coverage, radial_alignment = circle_edge_support(edges, gradient_x, gradient_y, x, y, r)
    large_weak_penalty = 0.25 if r > 130 and density < 0.19 else 0
    score = (1.2 * density) + (0.7 * coverage) + (0.9 * radial_alignment) - large_weak_penalty
    return score


def select_dense_candidates(gray, edges, candidates):
    gradient_x = cv2.Sobel(gray.astype("float32"), cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray.astype("float32"), cv2.CV_32F, 0, 1, ksize=3)
    scored = []

    for x, y, r in candidates:
        if not is_valid_coin(gray, edges, x, y, r):
            continue

        score = score_circle_candidate(edges, gradient_x, gradient_y, x, y, r)
        if score >= 1.45:
            scored.append((score, x, y, r))

    selected = []

    for score, x, y, r in sorted(scored, reverse=True):
        keep = True

        for sx, sy, sr in selected:
            center_dist = np.sqrt((x - sx) ** 2 + (y - sy) ** 2)
            same_circle = (
                center_dist <= 0.65 * min(r, sr)
                and abs(r - sr) <= 0.35 * max(r, sr)
            )

            if same_circle:
                keep = False
                break

        if keep:
            selected.append((x, y, r))

    return selected


def is_valid_coin(gray, edges, x, y, r, dense=False, median_radius=None):
    height, width = gray.shape
    margin = max(2, int(r * 0.08))

    if x - r < -margin or y - r < -margin or x + r >= width + margin or y + r >= height + margin:
        return False

    coin_mask = np.zeros(gray.shape, dtype="uint8")
    cv2.circle(coin_mask, (x, y), int(r * 0.88), 255, -1)

    mean_intensity, std_intensity = cv2.meanStdDev(gray, mask=coin_mask)
    edge_density = circle_edge_density(edges, x, y, r)

    if dense and median_radius is not None and r >= 0.85 * median_radius and edge_density < 0.14:
        return False

    # Coins normally have texture inside and a visible circular boundary.
    return std_intensity[0][0] > 10 and edge_density > 0.08


def refine_small_coin_radii(edges, circles):
    if len(circles) == 0:
        return circles

    median_radius = np.median([r for _, _, r in circles])
    refined = []

    for x, y, r in circles:
        if r > 0.75 * median_radius:
            refined.append((x, y, r))
            continue

        best_r = r
        best_score = -1
        start = max(8, int(r * 0.85))
        stop = max(start + 1, int(r * 1.10))

        for test_r in range(start, stop + 1):
            score = circle_edge_density(edges, x, y, test_r)
            if score > best_score:
                best_score = score
                best_r = test_r

        refined.append((x, y, best_r))

    return refined


def remove_contained_circles(circles):
    filtered = []

    for circle in sorted(circles, key=lambda item: item[2], reverse=True):
        x, y, r = circle
        contained = False

        for fx, fy, fr in filtered:
            dist = np.sqrt((x - fx) ** 2 + (y - fy) ** 2)
            if dist < 0.95 * fr and r > 0.75 * fr:
                contained = True
                break

        if not contained:
            filtered.append(circle)

    return filtered


def remove_nested_dense_circles(circles):
    filtered = []

    for circle in sorted(circles, key=lambda item: item[2], reverse=True):
        x, y, r = circle
        nested = False

        for fx, fy, fr in filtered:
            dist = np.sqrt((x - fx) ** 2 + (y - fy) ** 2)
            same_coin = dist < 0.95 * fr and r > 0.75 * fr
            inner_artifact = dist < 0.78 * fr and r < 0.50 * fr

            if same_coin or inner_artifact:
                nested = True
                break

        if not nested:
            filtered.append((int(x), int(y), int(r)))

    return filtered



# Extract HSV color features 3ashan n2dar n3mel classify bronze/silver/gold
def extract_color_features(image, circles):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    feats = []

    for (x, y, r) in circles:
        mask = np.zeros(image.shape[:2], dtype="uint8")

        # slightly smaller mask avoids edge reflections
        cv2.circle(mask, (x, y), int(r * 0.85), 255, -1)

        h, s, v, _ = cv2.mean(hsv, mask=mask)
        feats.append([h, s, v])

    return np.array(feats)


# Classify material using HSV -- > bronze/silver/gold

def classify_materials(color_feats):
    material_to_label = {"Gold": 0, "Silver": 1, "Bronze": 2}
    color_names = {0: "Gold", 1: "Silver", 2: "Bronze"}
    labels = []

    for h, s, v in color_feats:
        # Hand-written HSV rules: no trained model, just color thresholds.
        if s < 85 and v > 115:
            material = "Silver"
        elif 15 <= h <= 45 and s > 80 and v > 130:
            material = "Gold"
        else:
            material = "Bronze"

        labels.append(material_to_label[material])

    return np.array(labels), color_names


def sort_circles(circles):
    return sorted(circles, key=lambda circle: (circle[1], circle[0]))


def draw_visible_arc(canvas, edges, x, y, r, color):
    ring_outer = np.zeros(edges.shape, dtype="uint8")
    ring_inner = np.zeros(edges.shape, dtype="uint8")
    cv2.circle(ring_outer, (x, y), r + 3, 255, -1)
    cv2.circle(ring_inner, (x, y), max(1, r - 4), 255, -1)
    ring = cv2.subtract(ring_outer, ring_inner)
    visible = cv2.bitwise_and(edges, edges, mask=ring)
    visible = cv2.dilate(visible, np.ones((2, 2), dtype="uint8"), iterations=1)
    canvas[visible > 0] = color


def draw_coin_results(image, coin_results, total, edges=None, dense=False):
    output = image.copy()
    height, width = output.shape[:2]
    panel_width = 280 if total > 25 else 250
    canvas = np.full((height, width + panel_width, 3), 255, dtype=np.uint8)
    canvas[:, :width] = output

    cv2.putText(
        canvas,
        f"Coins: {total}",
        (width + 16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (30, 30, 30),
        2,
    )

    circle_color = (0, 160, 255)
    circle_thickness = 1 if total > 25 else 2
    badge_radius = 9 if total > 25 else 12
    badge_font = 0.35 if total > 25 else 0.45
    row_start = 52 if total > 25 else 66
    row_step = 18 if total > 25 else 24
    row_font = 0.4 if total > 25 else 0.5

    for coin in coin_results:
        idx = coin["id"]
        x, y, r = coin["circle"]
        material = coin["material"]
        size = coin["size"]

        if not dense:
            cv2.circle(canvas, (x, y), r, circle_color, circle_thickness)

        cv2.circle(canvas, (x, y), badge_radius, circle_color, -1)
        cv2.circle(canvas, (x, y), badge_radius, (255, 255, 255), 1)
        cv2.putText(
            canvas,
            str(idx),
            (x - 5 if idx < 10 else x - 8, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            badge_font,
            (20, 20, 20),
            1 if total > 25 else 2,
        )

        row_y = row_start + (idx - 1) * row_step
        if row_y < height - 12:
            cv2.putText(
                canvas,
                f"{idx}. {material} - {size}",
                (width + 18, row_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                row_font,
                (30, 30, 30),
                1,
            )

    return canvas



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

    output = image.copy()

    min_radius, max_radius = estimate_radius_range(image.shape, min_radius, max_radius)
    if mode == "auto":
        dense = min(image.shape[:2]) >= 650
    else:
        dense = mode == "dense"

    gray, blur, edges = preprocess_image(image, normalize=not dense and min(image.shape[:2]) >= 500)
    circles = find_circle_candidates(blur, min_radius, max_radius, dense=dense)

    if debug:
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        print(f"Radius range: {min_radius}-{max_radius}px")
        print(f"Mode: {'dense' if dense else 'balanced'}")
        print(f"Hough candidates: {len(circles)}")

    if len(circles) == 0:
        print("No circles detected")
        return None

    valid_circles = []

    if dense:
        valid_circles = select_dense_candidates(gray, edges, circles)
    else:
        candidate_median_radius = np.median(circles[:, 2]) if len(circles) > 0 else None

        for (x, y, r) in circles:
            if is_valid_coin(gray, edges, x, y, r, dense=dense, median_radius=candidate_median_radius):

                # expand circle slightly to match coin edge
                r = int(r * 1.03)

                valid_circles.append((x, y, r))

        valid_circles = remove_contained_circles(valid_circles)

    if dense:
        valid_circles = [(x, y, int(r * 1.03)) for x, y, r in valid_circles]
        valid_circles = remove_nested_dense_circles(valid_circles)

    valid_circles = refine_small_coin_radii(edges, valid_circles)

    if len(valid_circles) == 0:
        print("No valid coins")
        return None


    # classification bel radius
    valid_circles = sort_circles(valid_circles)
    radii = np.array([r for (_, _, r) in valid_circles])
    median_r = np.median(radii)


    # classification bel color
    color_feats = extract_color_features(image, valid_circles)
    labels, color_names = classify_materials(color_feats)


    # Counters
    total = len(valid_circles)

    material_counts = {"Gold":0, "Silver":0, "Bronze":0}
    size_counts = {"Big":0, "Small":0}
    combo_counts = {
        "Gold_Big":0, "Gold_Small":0,
        "Silver_Big":0, "Silver_Small":0,
        "Bronze_Big":0, "Bronze_Small":0
    }


    coin_results = []

    for i, (x, y, r) in enumerate(valid_circles):
        material = color_names[labels[i]]
        size = "Big" if r >= median_r else "Small"

        material_counts[material] += 1
        size_counts[size] += 1
        combo_counts[f"{material}_{size}"] += 1

        coin_results.append(
            {
                "id": i + 1,
                "circle": (x, y, r),
                "material": material,
                "size": size,
            }
        )


    print("\nTotal coins:", total)

    print("\nBy material:")
    print(material_counts)

    print("\nBy size:")
    print(size_counts)

    print("\nDetailed:")
    print(combo_counts)

    output = draw_coin_results(image, coin_results, total, edges=edges, dense=dense)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output)
    print(f"\nSaved annotated image to: {output_path}")

    if show:
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"Coins detected: {total}")
        plt.show()

    return {
        "total": total,
        "material_counts": material_counts,
        "size_counts": size_counts,
        "combo_counts": combo_counts,
        "output_path": str(output_path),
        "radius_range": (min_radius, max_radius),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Detect and classify coins in a local image.")
    parser.add_argument(
        "image",
        nargs="?",
        default="images/3.jpg",
        help="Path to the input coin image.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path where the annotated image will be saved.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the result without opening a matplotlib window.",
    )
    parser.add_argument(
        "--min-radius",
        type=int,
        default=None,
        help="Optional minimum coin radius in pixels.",
    )
    parser.add_argument(
        "--max-radius",
        type=int,
        default=None,
        help="Optional maximum coin radius in pixels.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detection parameters for tuning a new image.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "balanced", "dense"),
        default="auto",
        help="Detection mode. Dense favors recall on crowded coin piles.",
    )
    return parser.parse_args()


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
