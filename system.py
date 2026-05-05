import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Remove duplicate circles
def remove_overlapping_circles(circles):
    if len(circles) == 0:
        return circles

    circles = sorted(circles, key=lambda x: x[2], reverse=True)
    filtered = []

    for (x, y, r) in circles:
        keep = True
        for (fx, fy, fr) in filtered:
            dist = np.sqrt((x - fx)**2 + (y - fy)**2)

            if dist < 0.7 * max(r, fr):
                keep = False
                break

        if keep:
            filtered.append((x, y, r))

    return np.array(filtered)



# Validate detected circle

def is_valid_coin(gray, x, y, r):
    mask = np.zeros(gray.shape, dtype="uint8")
    cv2.circle(mask, (x, y), r, 255, -1)

    mean_intensity = cv2.mean(gray, mask=mask)[0]
    return 60 < mean_intensity < 220



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
    n_clusters = 3 if len(color_feats) >= 3 else 2

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(color_feats)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    color_names = {}

    for i, (h, s, v) in enumerate(centers):

        # - Gold → high saturation
        # - Silver → low saturation, high value
        # - Bronze → medium saturation, darker

        if s > 100 and v > 120:
            color_names[i] = "Gold"
        elif s < 60 and v > 120:
            color_names[i] = "Silver"
        else:
            color_names[i] = "Bronze"

    return labels, color_names



def detect_coins(image_path):
    image = cv2.imread(image_path)
    output = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 1.5)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=90,
        param1=120,
        param2=45,
        minRadius=35,
        maxRadius=80
    )

    if circles is None:
        print("No circles detected")
        return

    circles = np.round(circles[0, :]).astype("int")
    circles = remove_overlapping_circles(circles)

    valid_circles = []

    for (x, y, r) in circles:
        if is_valid_coin(gray, x, y, r):

            # expand circle slightly to match coin edge
            r = int(r * 1.052)

            valid_circles.append((x, y, r))

    if len(valid_circles) == 0:
        print("No valid coins")
        return


    # classification bel radius
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


    for i, (x, y, r) in enumerate(valid_circles):
        material = color_names[labels[i]]
        size = "Big" if r > median_r else "Small"

        material_counts[material] += 1
        size_counts[size] += 1
        combo_counts[f"{material}_{size}"] += 1

        color = (0,215,255) if material=="Gold" else \
                (200,200,200) if material=="Silver" else \
                (42,42,165)

        cv2.circle(output, (x, y), r, color, 2)

        cv2.putText(output,
                    f"{material}-{size}",
                    (x - 40, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color, 2)


    print("\nTotal coins:", total)

    print("\nBy material:")
    print(material_counts)

    print("\nBy size:")
    print(size_counts)

    print("\nDetailed:")
    print(combo_counts)


    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Coins detected: {total}")
    plt.show()


if __name__ == "__main__":
    detect_coins("images/coins3.jpg")