import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import linregress
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score

def load_filenames_from_txt(file_path):
    with open(file_path, 'r') as file:
        filenames = [line.strip() for line in file]
    return filenames


def load_image_shape(image_name):
    image_path = os.path.join(images_path, image_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image.shape if image is not None else None


def load_coordinates(file_path):
    return np.loadtxt(file_path)


def connect_ends_and_create_mask(lumen_coords, media_coords, image_shape):

    lumen_mask = np.zeros(image_shape, dtype=np.uint8)
    media_mask = np.zeros(image_shape, dtype=np.uint8)

    lumen_coords = lumen_coords.astype(np.int32)
    media_coords = media_coords.astype(np.int32)

    cv2.polylines(lumen_mask, [lumen_coords], isClosed=False, color=1, thickness=1)
    cv2.polylines(media_mask, [media_coords], isClosed=False, color=1, thickness=1)

    contour_coords = np.vstack((lumen_coords, media_coords[::-1]))
    tube_mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(tube_mask, [contour_coords], color=1)

    return tube_mask


def process_masks_from_images(txt_file_path, lima_profiles_path, target_size=(256, 256)):
    image_filenames = load_filenames_from_txt(txt_file_path)
    masks = []

    for image_name in image_filenames:
        image_shape = load_image_shape(image_name)
        if image_shape is None:
            continue

        base_name = os.path.splitext(image_name)[0]
        lumen_file = os.path.join(lima_profiles_path, f"{base_name}-LI.txt")
        media_file = os.path.join(lima_profiles_path, f"{base_name}-MA.txt")

        if not (os.path.exists(lumen_file) and os.path.exists(media_file)):
            continue

        lumen_coords = load_coordinates(lumen_file)
        media_coords = load_coordinates(media_file)

        if lumen_coords is not None and media_coords is not None:
            tube_mask = connect_ends_and_create_mask(lumen_coords, media_coords, image_shape)
            # Resize the mask to target size
            tube_mask_resized = cv2.resize(tube_mask, target_size, interpolation=cv2.INTER_NEAREST)
            masks.append(tube_mask_resized)

    return masks


def max_average_vertical_distance(contour, window_size=5):
    """
    Calculate the maximum average vertical span within a sliding window along the x-axis.

    Parameters:
    - contour: List of contour points, where each point is in the form of [[x, y]].
    - window_size: Number of x-coordinates to include in each window.

    Returns:
    - Maximum average vertical span across all windows.
    """
    if contour is None or len(contour) == 0:
        return 0

    # Dictionary to hold y-coordinates grouped by x
    vertical_spans = {}

    for pt in contour:
        x, y = pt[0]
        if x in vertical_spans:
            vertical_spans[x].append(y)
        else:
            vertical_spans[x] = [y]

    x_values = sorted(vertical_spans.keys())
    max_avg_span = 0

    # Iterate over x_values with a window
    for i in range(len(x_values) - window_size + 1):
        current_window = x_values[i:i + window_size]
        total_span = 0
        valid_lines = 0

        for x in current_window:
            y_list = vertical_spans[x]
            if len(y_list) >= 2:
                total_span += max(y_list) - min(y_list)
                valid_lines += 1

        if valid_lines > 0:
            avg_span = total_span / valid_lines
            max_avg_span = max(max_avg_span, avg_span)

    return max_avg_span


def extract_main_contour(mask):

    binary_mask = (mask > 0.25).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour
    return None


def load_conversion_factor(file_path):
    with open(file_path, 'r') as file:
        factor = float(file.read().strip())
    return factor


def extract_distances_from_masks(masks, filenames, conversion_factors_path, conversion_file_suffix='_CF.txt', window_size=5):
    distances = {}

    for idx, mask in enumerate(masks):

        main_contour = extract_main_contour(mask)
        image_filename = filenames[idx].strip('.tiff')
        if main_contour is None:
            print(f"No main contour found for mask {idx}.")
            distances[idx] = None
            continue

        max_avg_distance = max_average_vertical_distance(main_contour, window_size)

        conversion_factor_file = os.path.join(conversion_factors_path, f"{image_filename}{conversion_file_suffix}")
        if os.path.exists(conversion_factor_file):
            pixel_to_mm = load_conversion_factor(conversion_factor_file)
            max_avg_distance_mm = max_avg_distance * pixel_to_mm
            distances[idx] = max_avg_distance_mm
        else:
            print(f"Conversion factor file not found for mask {idx}.")
            distances[idx] = None

    return distances


pred_masks = np.load('./data/predicted_masks_v2.npy')
conversion_factors_path = './CUBS2/CF'

images_path = "./CUBS2/IMAGES"
lima_profiles_path = "./CUBS2/SEGMENTATION/A1"
txt_file_path = "./data/validation_filenames.txt"
filenames = load_filenames_from_txt(txt_file_path)

true_masks = process_masks_from_images(txt_file_path, lima_profiles_path)

true_distances = extract_distances_from_masks(true_masks, filenames, conversion_factors_path)

pred_distances = extract_distances_from_masks(pred_masks, filenames, conversion_factors_path)


for idx in range(len(pred_masks)):
    pred_dist = pred_distances.get(idx)
    true_dist = true_distances.get(idx)

    print(f"Image {idx}: Predicted Distance: {pred_dist} mm, True Distance: {true_dist} mm")


def plot_mask_comparison_with_image(original_images_path, pred_masks, true_masks, indices, num_samples=5,
                                    target_size=(256, 256)):
    """
    Plot comparison of original images, predicted masks, and true masks for given indices.

    Parameters:
    - original_images_path: Path to the folder containing original images.
    - pred_masks: Array of predicted masks.
    - true_masks: Array of true masks.
    - indices: List of indices to visualize.
    - num_samples: Number of samples to plot (default is 5).
    - target_size: Target size to which the original image will be resized.
    """
    num_samples = min(num_samples, len(indices))
    sampled_indices = indices[:num_samples]

    for idx in sampled_indices:
        plt.figure(figsize=(15, 5))

        image_name = os.path.basename(filenames[idx])
        original_image_path = os.path.join(original_images_path, image_name)
        original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        original_image_resized = cv2.resize(original_image, target_size)

        plt.subplot(1, 3, 1)
        plt.imshow(original_image_resized, cmap='gray')
        plt.title(f'Original Image {image_name}')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(pred_masks[idx], cmap='gray')
        plt.title(f'Predicted Mask {idx}')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(true_masks[idx], cmap='gray')
        plt.title(f'True Mask {idx}')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


indices_to_visualize = [431, 432, 433, 434, 435]


def compute_specificity_sensitivity_for_thresholds(pred_distances, true_distances, cimt_thresholds):
    """
    Compute specificity and sensitivity for multiple thresholds.

    Parameters:
    - pred_distances: Dictionary of predicted distances.
    - true_distances: Dictionary of true distances.
    - cimt_thresholds: List or array of thresholds to evaluate.

    Returns:
    - sens_spec: Dictionary with thresholds as keys and (sensitivity, specificity) tuples as values.
    """
    sens_spec = {}

    for cimt in cimt_thresholds:
        TP = FP = TN = FN = 0

        for identifier, pred_dist in pred_distances.items():
            actual_dist = true_distances.get(identifier, 0)
            pred_positive = pred_dist > cimt
            actual_positive = actual_dist > cimt

            if pred_positive and actual_positive:
                TP += 1
            elif pred_positive and not actual_positive:
                FP += 1
            elif not pred_positive and not actual_positive:
                TN += 1
            elif not pred_positive and actual_positive:
                FN += 1

        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        sens_spec[cimt] = (sensitivity, specificity)

    return sens_spec


def plot_roc_curves_for_thresholds(pred_distances, true_distances, cimt_thresholds):
    """
    Plot ROC curves for multiple CIMT thresholds on the same figure.

    Parameters:
    - pred_distances: Dictionary of predicted distances.
    - true_distances: Dictionary of true distances.
    - cimt_thresholds: List or array of thresholds to evaluate.

    Returns:
    - sens_spec: Dictionary capturing sensitivity and specificity for each threshold.
    """
    sens_spec = compute_specificity_sensitivity_for_thresholds(pred_distances, true_distances, cimt_thresholds)

    plt.figure(figsize=(10, 8))

    for cimt in cimt_thresholds:
        y_true = np.array([true_distances.get(identifier, 0) > cimt for identifier in pred_distances.keys()])
        y_scores = np.array(list(pred_distances.values()))

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f'Threshold {cimt}, AUC = {roc_auc:.2f}')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificitás (Hamis pozitív ráta)')
    plt.ylabel('Szenzitivitás (Valós pozitív ráta)')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.show()

    return sens_spec


cimt_threshold = 0.8

cimt_values = np.linspace(0.8, 0.8, 1)
sens_spec = plot_roc_curves_for_thresholds(pred_distances, true_distances, cimt_values)



for threshold, (sensitivity, specificity) in sens_spec.items():
    print(f"Threshold {threshold:.2f}: Sensitivity = {sensitivity:.2f}, Specificity = {specificity:.2f}")


def plot_distances(pred_distances, true_distances):
    """
    Plot predicted distances vs. true distances.

    Parameters:
    - pred_distances: Dictionary with predicted distances.
    - true_distances: Dictionary with true distances.
    """
    common_keys = set(pred_distances.keys()).intersection(true_distances.keys())
    pred_values = [pred_distances[key] for key in common_keys]
    true_values = [true_distances[key] for key in common_keys]

    plt.figure(figsize=(8, 8))
    plt.scatter(pred_values, true_values, alpha=0.7, edgecolor='k')

    min_value = min(pred_values + true_values)
    max_value = max(pred_values + true_values)
    plt.plot([min_value, max_value], [min_value, max_value], 'r--', lw=2, label='y = x')

    plt.xlabel('Prediktált CIMT (mm)')
    plt.ylabel('Valódi CIMT (mm)')
    plt.title('Prediktált vs. Valódi CIMT')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()


plot_distances(pred_distances, true_distances)


def calculate_segmentation_metrics(true_masks, pred_masks, threshold=0.5):
    """Calculate averaged Dice, Jaccard, Precision, Recall over all masks.

    Parameters:
    - true_masks: List or array of true binary masks.
    - pred_masks: List or array of predicted masks (might require thresholding).
    - threshold: Threshold to binarize predicted masks.

    Returns:
    - Averaged metrics dictionary.
    """
    dice_scores = []
    jaccard_scores = []
    precision_scores = []
    recall_scores = []

    for true_mask, pred_mask in zip(true_masks, pred_masks):

        pred_mask_binary = (pred_mask >= threshold).astype(np.uint8)

        true_flat = true_mask.flatten()
        pred_flat = pred_mask_binary.flatten()

        dice = f1_score(true_flat, pred_flat, average='binary')
        jaccard = jaccard_score(true_flat, pred_flat, average='binary')
        precision = precision_score(true_flat, pred_flat, average='binary')
        recall = recall_score(true_flat, pred_flat, average='binary')

        dice_scores.append(dice)
        jaccard_scores.append(jaccard)
        precision_scores.append(precision)
        recall_scores.append(recall)

    metrics = {
        "Average Dice": np.mean(dice_scores),
        "Average Jaccard (IoU)": np.mean(jaccard_scores),
        "Average Precision": np.mean(precision_scores),
        "Average Recall": np.mean(recall_scores)
    }

    return metrics


metrics = calculate_segmentation_metrics(true_masks, pred_masks)
print(metrics)
