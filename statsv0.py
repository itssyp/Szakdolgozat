import os
import numpy as np
import cv2
from scipy.spatial import distance
import matplotlib.pyplot as plt
import json

# Load the predicted mask
pred = np.load('./data/predicted_masks.npy')
conversion_factors_path = './CUBS2/CF'


def connect_ends_and_create_mask(lumen_coords, media_coords, image_shape):
    # Create empty masks for lumen and media
    lumen_mask = np.zeros(image_shape, dtype=np.uint8)
    media_mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert lines to integer coordinates if necessary
    lumen_coords = lumen_coords.astype(np.int32)
    media_coords = media_coords.astype(np.int32)

    # Draw the lines for lumen-intima and media-adventitia
    cv2.polylines(lumen_mask, [lumen_coords], isClosed=False, color=1, thickness=1)
    cv2.polylines(media_mask, [media_coords], isClosed=False, color=1, thickness=1)

    # Stack the coordinates to form a closed polygon
    contour_coords = np.vstack((lumen_coords, media_coords[::-1]))  # Reverse the order of media to ensure they join correctly

    # Create the tube mask
    tube_mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(tube_mask, [contour_coords], color=1)

    return tube_mask


def extract_main_contour(mask):
    # Convert mask to binary
    binary_mask = (mask > 0.15).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Assume the largest contour is the artery wall
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour
    return None


def max_vertical_distance(contour):
    if contour is None:
        return 0

    vertical_spans = {}

    for pt in contour:
        x, y = pt[0]  # pt is usually in the form of [[x, y]]

        # Group all y-coordinates by x-coordinate
        if x in vertical_spans:
            vertical_spans[x].append(y)
        else:
            vertical_spans[x] = [y]

    # Calculate the maximum vertical distance for each x and track maximum distance
    max_span = 0
    for x, y_list in vertical_spans.items():
        if len(y_list) >= 2:
            current_span = max(y_list) - min(y_list)
            if current_span > max_span:
                max_span = current_span

    return max_span


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


def max_perpendicular_distance(lumen_contour, media_contour):
    # Extract x, y coordinates from contours
    lumen_pts = lumen_contour[:, 0, :]
    media_pts = media_contour[:, 0, :]

    # Sort points based on x-coordinates
    lumen_sorted = sorted(lumen_pts, key=lambda pt: pt[0])
    media_sorted = sorted(media_pts, key=lambda pt: pt[0])

    max_vertical_dist = 0

    # Calculate vertical distance for each x point
    for lx, ly in lumen_sorted:
        # Find corresponding media point with the same x
        media_pairs = [pt for pt in media_sorted if pt[0] == lx]

        if media_pairs:
            _, my = media_pairs[0]
            vertical_dist = abs(my - ly)

            if vertical_dist > max_vertical_dist:
                max_vertical_dist = vertical_dist

    return max_vertical_dist


def load_conversion_factor(file_path):
    with open(file_path, 'r') as file:
        # Assuming the file contains a single float value representing the conversion factor
        factor = float(file.read().strip())
    return factor


def plot_contour(image, contour):
    # Create a copy of the image to draw on
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if contour is not None:
        # Draw contour
        cv2.drawContours(overlay, [contour], -1, (255, 0, 0), 1)  # Blue for the contour

    # Plot the image and contours
    plt.figure(figsize=(10, 5))
    plt.imshow(overlay)
    plt.title('Artery Wall Contour')
    plt.axis('off')
    plt.show()


# Iterate over each predicted mask for testing/debugging
def debug_plot(pred):
    for idx, pred_mask in enumerate(pred):
        pred_mask_squeezed = pred_mask.squeeze()
        main_contour = extract_main_contour(pred_mask_squeezed)

        # Use the mask for plotting
        plot_contour(pred_mask_squeezed, main_contour)

        # Break after the first plot for demonstration; remove for full iteration
        break



def load_filenames_from_txt(file_path):
    """
    Load filenames from a text file.
    """
    with open(file_path, 'r') as file:
        filenames = [line.strip() for line in file]
    return filenames



def load_history_from_json(filename):
    # Load JSON data back into a dictionary
    with open(filename, 'r') as f:
        history_dict = json.load(f)
    return history_dict


def plot_comparison(history_data_bce, history_data_combined, metric='loss'):
    plt.figure(figsize=(10, 5))

    # Plot for Binary Cross-Entropy
    plt.plot(history_data_bce[metric], label='BCE Training')
    plt.plot(history_data_bce[f'val_{metric}'], label='BCE Validation')

    # Plot for Combined Dice and BCE
    plt.plot(history_data_combined[metric], label='Combined Training')
    plt.plot(history_data_combined[f'val_{metric}'], label='Combined Validation')

    plt.title(f'Comparison of {metric.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.show()


def scale_coordinates(coords, original_shape, target_shape=(256, 256)):
    """
    Scale coordinates from the original image shape to the target shape.
    """
    scale_x = target_shape[1] / original_shape[1]
    scale_y = target_shape[0] / original_shape[0]
    scaled_coords = coords.copy()
    scaled_coords[:, 0] *= scale_x  # Scale x coordinates
    scaled_coords[:, 1] *= scale_y  # Scale y coordinates
    return scaled_coords


def load_image_shape(image_path):
    """Load an image to determine its shape."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image.shape if image is not None else (None, None)


def load_coordinates_for_filenames(base_path, image_names):
    """Load and return coordinates for each image given the naming convention."""
    profiles = {}
    for image_name in image_names:
        base_name = os.path.splitext(image_name)[0]
        lumen_file = os.path.join(base_path, f"{base_name}-LI.txt")
        media_file = os.path.join(base_path, f"{base_name}-MA.txt")

        if os.path.exists(lumen_file) and os.path.exists(media_file):
            lumen_coords = np.loadtxt(lumen_file)
            media_coords = np.loadtxt(media_file)
            profiles[base_name] = (lumen_coords, media_coords)

    return profiles


def calculate_distances(profiles, original_shapes, target_shape=(256, 256), window_size=1):
    """Compute the maximum average vertical distance for each profile."""
    distances = {}
    for idx, (key, (lumen_coords, media_coords)) in enumerate(profiles.items()):
        scaled_lumen = scale_coordinates(lumen_coords, original_shapes[idx], target_shape)
        scaled_media = scale_coordinates(media_coords, original_shapes[idx], target_shape)

        tube_mask = connect_ends_and_create_mask(scaled_lumen, scaled_media, target_shape)
        contour = extract_main_contour(tube_mask)

        if contour is not None:
            distance = max_average_vertical_distance(contour, window_size)
            distances[key] = distance
        else:
            distances[key] = None  # Handle cases with invalid contour
    return distances


# Define paths to different datasets for comparison
A1_path = "./CUBS2/SEGMENTATION/A1"
A2_path = "./CUBS2/SEGMENTATION/Manual-A2"
A3_path = "path/to/computer/lima_profiles"
images_path = "./CUBS2/IMAGES"
filenames = load_filenames_from_txt('./data/validation_filenames.txt')


# history_bce_data = load_history_from_json('history_bce.json')

# Plot comparison
# plot_comparison(history_bce_data, history_bce_data, 'loss')
# plot_comparison(history_bce_data, history_bce_data, 'accuracy')

# Derive original shapes from the images
original_shapes = [load_image_shape(os.path.join(images_path, filename)) for filename in filenames]


# Load and calculate distances for these profiles
A1_profiles = load_coordinates_for_filenames(A1_path, filenames)
A1_distances = calculate_distances(A1_profiles, original_shapes)

A2_profiles = load_coordinates_for_filenames(A2_path, filenames)
A2_distances = calculate_distances(A2_profiles, original_shapes)



# Print calculated distances
# for name in filenames:
#     base_name = os.path.splitext(name)[0]
#     # Construct the path for conversion factor file related to the image
#     conversion_factor_file = os.path.join(conversion_factors_path, f"{os.path.splitext(base_name)[0]}_CF.txt")
#     max_dist_pixels = A2_distances.get(base_name)
#
#     if os.path.exists(conversion_factor_file):
#         pixel_to_mm = load_conversion_factor(conversion_factor_file)
#         max_dist_mm = max_dist_pixels * pixel_to_mm
#         print(f"Image: {base_name}")
#         print(f"Distance: {max_dist_mm}")

def stats(pred, A2_distances, filenames, conversion_factors_path, output_file):
    """
    Compare predicted distances against A2 distances for a given list of filenames.

    Parameters:
    - pred: List of predicted masks.
    - A2_distances: Dictionary containing the A2 distances by filename.
    - filenames: List of image filenames corresponding to predictions.
    - conversion_factors_path: Path where conversion factors are stored.
    """
    results = {}
    for idx, pred_mask in enumerate(pred):
        # Use the filenames read from file
        image_filename = filenames[idx]
        pred_mask_squeezed = pred_mask.squeeze()
        main_contour = extract_main_contour(pred_mask_squeezed)
        print(image_filename)
        if main_contour is not None:
            pred_max_dist_pixels = max_average_vertical_distance(main_contour)
            A2_max_dist_pixels = A2_distances.get(image_filename.strip('.tiff'), 0)

            # Construct the path for conversion factor file related to the image
            conversion_factor_file = os.path.join(conversion_factors_path,
                                                  f"{os.path.splitext(image_filename)[0]}_CF.txt")

            if os.path.exists(conversion_factor_file):
                pixel_to_mm = load_conversion_factor(conversion_factor_file)
                pred_max_dist_mm = pred_max_dist_pixels * pixel_to_mm
                A2_max_dist_mm = A2_max_dist_pixels * pixel_to_mm
                print(
                    f"Image {image_filename}: Pred: {pred_max_dist_mm:.2f} mm "
                    f"A2: {A2_max_dist_mm:.2f} mm"
                    f"(Conversion factor: {pixel_to_mm:.4f} mm/px) "
                )

                # Save results to the dictionary
                results[image_filename] = {
                    'pred_max_dist_mm': pred_max_dist_mm,
                    'A2_max_dist_mm': A2_max_dist_mm
                }
            else:
                print(f"Conversion factor file for Image {image_filename} not found.")
        else:
            print(f"No contour found for Image {image_filename}.")
# Write results to a JSON file for later use
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)


# Example call
output_file_path = 'calculated_widths_v2.json'
stats(pred, A1_distances, filenames, conversion_factors_path, output_file_path)