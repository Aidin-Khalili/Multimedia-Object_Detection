import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.svm import SVC


def read_images(path, target_shape):
    image_list = []
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        image = Image.open(file_path) 
        if image.mode != "RGB":
            image = image.convert("RGB")
            print(f"{filename} converted to RGB format.")
        image = image.resize(target_shape)
        image = np.array(image)
        image_list.append(image)
    return np.array(image_list)

def present_images(image_list):
    for i in range(len(image_list)):
        image = image_list[i]
        if len(image.shape) == 3 and image.shape[2] == 3:
            plt.imshow(image)
            print("RGB image printed")
        else:
            plt.imshow(image, cmap="gray")
            print("Grayscale image printed")
        plt.axis("off")
        plt.show()

def save_images(image_list, path):
    for i, image in enumerate(image_list):
        filename = f"Pic-{i}.png"
        file_path = os.path.join(path, filename)
        if len(image.shape) == 2:
            img = Image.fromarray(image, mode='L')
            print("Grayscale image printed")
        else:
            img = Image.fromarray(image, mode='RGB')
            print("RGB image printed")
        img.save(file_path)

def calculate_histograms(image_list):
    histograms = []
    for image in image_list:
        histogram_red = [0] * 256
        histogram_green = [0] * 256
        histogram_blue = [0] * 256
        for row in image:
            for pixel in row:
                r, g, b = pixel
                histogram_red[r] += 1
                histogram_green[g] += 1
                histogram_blue[b] += 1
        histograms.append((histogram_red, histogram_green, histogram_blue))
    return histograms

def save_histograms(histograms, path):
    os.makedirs(path, exist_ok=True)  # Create the directory if it doesn't exist
    for i, (histogram_red, histogram_green, histogram_blue) in enumerate(histograms):
        # Create histogram images for each channel
        histogram_image_red = np.zeros((256, 256, 3), dtype=np.uint8)
        histogram_image_green = np.zeros((256, 256, 3), dtype=np.uint8)
        histogram_image_blue = np.zeros((256, 256, 3), dtype=np.uint8)
        histogram_image_red[:, :, 0] = np.interp(histogram_red, (0, np.max(histogram_red)), (0, 255)).astype(np.uint8)
        histogram_image_green[:, :, 1] = np.interp(histogram_green, (0, np.max(histogram_green)), (0, 255)).astype(np.uint8)
        histogram_image_blue[:, :, 2] = np.interp(histogram_blue, (0, np.max(histogram_blue)), (0, 255)).astype(np.uint8)

        histogram_red_name = f"Histogram-Red-{i+1}.png"
        histogram_green_name = f"Histogram-Green-{i+1}.png"
        histogram_blue_name = f"Histogram-Blue-{i+1}.png"
        histogram_red_path = os.path.join(path, histogram_red_name)
        histogram_green_path = os.path.join(path, histogram_green_name)
        histogram_blue_path = os.path.join(path, histogram_blue_name)

        # Save the histogram images for each channel
        histogram_red_pil = Image.fromarray(histogram_image_red)
        histogram_red_pil.save(histogram_red_path)

        histogram_green_pil = Image.fromarray(histogram_image_green)
        histogram_green_pil.save(histogram_green_path)

        histogram_blue_pil = Image.fromarray(histogram_image_blue)
        histogram_blue_pil.save(histogram_blue_path)

def histogram_equalization(image_list, histograms):
    equalized_images = []
    for i, (histogram_red, histogram_green, histogram_blue) in enumerate(histograms):
        image = image_list[i]
        height, width, channels = image.shape
        total_pixels = height * width
        cdf_red = [sum(histogram_red[:i+1]) for i in range(256)]
        cdf_green = [sum(histogram_green[:i+1]) for i in range(256)]
        cdf_blue = [sum(histogram_blue[:i+1]) for i in range(256)]
        cdf_red = np.array(cdf_red) / total_pixels
        cdf_green = np.array(cdf_green) / total_pixels
        cdf_blue = np.array(cdf_blue) / total_pixels
        equalized_image = np.zeros_like(image, dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                pixel = image[y, x]
                red = pixel[0]
                green = pixel[1]
                blue = pixel[2]
                equalized_red = int(cdf_red[red] * 255)
                equalized_green = int(cdf_green[green] * 255)
                equalized_blue = int(cdf_blue[blue] * 255)
                equalized_image[y, x] = [equalized_red, equalized_green, equalized_blue]
        equalized_images.append(equalized_image)
    return equalized_images

def laplacian_filter(image_list):
    enhanced_images = []
    for image in image_list:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        laplacian_rgb = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)
        enhanced_image = cv2.add(image, laplacian_rgb)
        enhanced_images.append(enhanced_image)
    return enhanced_images

def unsharp_masking(image_list, sigma=1.0, strength=1.5):
    enhanced_images = []
    for image in image_list:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
        high_pass = gray.astype(np.float32) - blurred.astype(np.float32)
        scaled_high_pass = high_pass * strength
        enhanced_gray = gray + scaled_high_pass
        enhanced_gray = np.clip(enhanced_gray, 0, 255).astype(np.uint8)
        enhanced_image = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
        enhanced_images.append(enhanced_image)
    return enhanced_images

def kmeans_segmentation(image_list, num_clusters):
    segmented_images = []
    for image in image_list:
        pixels = image.reshape(-1, 3)
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_pixels = centers[labels.flatten()]
        segmented_image = segmented_pixels.reshape(image.shape)
        segmented_images.append(segmented_image)
    return segmented_images

def gradient_edge_extraction(image_list):
    edge_images = []

    for image in image_list:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        gradient_magnitude_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        gradient_magnitude_rgb = cv2.cvtColor(gradient_magnitude_normalized, cv2.COLOR_GRAY2RGB)
        edge_images.append(gradient_magnitude_rgb)
    return edge_images

def polygonal_approximation(image_list, epsilon):
    approximated_images = []
    for image in image_list:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        approximations = []
        for contour in contours:
            epsilon = epsilon * cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(contour, epsilon, True)
            approximations.append(approximation)
        blank_image = np.zeros_like(image)
        cv2.drawContours(blank_image, approximations, -1, (255, 255, 255), 2)
        approximated_images.append(blank_image)
    return approximated_images

def detect_objects(image_list, min_object_size=1000):
    object_images = []
    object_masks = []
    compactness_list = []
    perimeter_list = []

    for image in image_list:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        filtered_labels = [label for label, stat in enumerate(stats[1:], start=1) if stat[4] > min_object_size]
        for label in filtered_labels:
            object_mask = np.where(labels == label, 255, 0).astype(np.uint8)
            contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            object_masks.append(object_mask)
            perimeter = cv2.arcLength(cnt, True)
            area = cv2.contourArea(cnt)
            compactness = 4 * np.pi * area / (perimeter ** 2)
            compactness_list.append(compactness)
            perimeter_list.append(perimeter)
        object_images.append(image)
    return object_images, object_masks, compactness_list, perimeter_list

def show_dataset(dataset):
    for data in dataset:
        compactness = data['compactness']
        perimeter = data['perimeter']
        label = data['label']
        print(f"Compactness: {compactness}, Perimeter: {perimeter}, Label: {label}")






image_path = "C:\\Users\\TamirLand\\Desktop\\Aidin\\Sadjad-Uni\\Lessons\\My-Terms\\Term 8\\Multimedia\\Image-part\\Project\\Pics\\Train-pics\\Orginal"
histogram_path = "C:\\Users\\TamirLand\\Desktop\\Aidin\\Sadjad-Uni\\Lessons\\My-Terms\\Term 8\\Multimedia\\Image-part\\Project\\Pics\\Train-pics\\Histogram"
enhancemented_path = "C:\\Users\\TamirLand\\Desktop\\Aidin\\Sadjad-Uni\\Lessons\\My-Terms\\Term 8\\Multimedia\\Image-part\\Project\\Pics\\Train-pics\\Enhancemented"
segmented_path = "C:\\Users\\TamirLand\\Desktop\\Aidin\\Sadjad-Uni\\Lessons\\My-Terms\\Term 8\\Multimedia\\Image-part\\Project\\Pics\\Train-pics\\Segmented"
edges_path = "C:\\Users\\TamirLand\\Desktop\\Aidin\\Sadjad-Uni\\Lessons\\My-Terms\\Term 8\\Multimedia\\Image-part\\Project\\Pics\\Train-pics\\Edges"
approximated_path = "C:\\Users\\TamirLand\\Desktop\\Aidin\\Sadjad-Uni\\Lessons\\My-Terms\\Term 8\\Multimedia\\Image-part\\Project\\Pics\\Train-pics\\Approximated"
object_path = "C:\\Users\\TamirLand\\Desktop\\Aidin\\Sadjad-Uni\\Lessons\\My-Terms\\Term 8\\Multimedia\\Image-part\\Project\\Pics\\Train-pics\\Object"

image_path2 = "C:\\Users\\TamirLand\\Desktop\\Aidin\\Sadjad-Uni\\Lessons\\My-Terms\\Term 8\\Multimedia\\Image-part\\Project\\Pics\\Train-pics\\Orginal2"
histogram_path2 = "C:\\Users\\TamirLand\\Desktop\\Aidin\\Sadjad-Uni\\Lessons\\My-Terms\\Term 8\\Multimedia\\Image-part\\Project\\Pics\\Train-pics\\Histogram2"
enhancemented_path2 = "C:\\Users\\TamirLand\\Desktop\\Aidin\\Sadjad-Uni\\Lessons\\My-Terms\\Term 8\\Multimedia\\Image-part\\Project\\Pics\\Train-pics\\Enhancemented2"
segmented_path2 = "C:\\Users\\TamirLand\\Desktop\\Aidin\\Sadjad-Uni\\Lessons\\My-Terms\\Term 8\\Multimedia\\Image-part\\Project\\Pics\\Train-pics\\Segmented2"
edges_path2 = "C:\\Users\\TamirLand\\Desktop\\Aidin\\Sadjad-Uni\\Lessons\\My-Terms\\Term 8\\Multimedia\\Image-part\\Project\\Pics\\Train-pics\\Edges2"
approximated_path2 = "C:\\Users\\TamirLand\\Desktop\\Aidin\\Sadjad-Uni\\Lessons\\My-Terms\\Term 8\\Multimedia\\Image-part\\Project\\Pics\\Train-pics\\Approximated2"
object_path2 = "C:\\Users\\TamirLand\\Desktop\\Aidin\\Sadjad-Uni\\Lessons\\My-Terms\\Term 8\\Multimedia\\Image-part\\Project\\Pics\\Train-pics\\Object2"

test_path = "C:\\Users\\TamirLand\\Desktop\\Aidin\\Sadjad-Uni\\Lessons\\My-Terms\\Term 8\\Multimedia\\Image-part\\Project\\Pics\\Train-pics\\Test"
test_object_path = "C:\\Users\\TamirLand\\Desktop\\Aidin\\Sadjad-Uni\\Lessons\\My-Terms\\Term 8\\Multimedia\\Image-part\\Project\\Pics\\Train-pics\\Tests'-Object"
########################################################Train part
#First For all data that we know there are screw

###0. Reading data from address
target_shape = (250, 250)
image_list = read_images(image_path, target_shape)
#image_list = laplacian_filter(image_list)

###1. Enhacement:
#  Computing histogram for equalization and storing it
histograms = calculate_histograms(image_list)
save_histograms(histograms, histogram_path)
histogram_equalized_images = histogram_equalization(image_list, histograms)

#  Using unsharp_masking
enhanced_images = unsharp_masking(histogram_equalized_images, sigma=1.0, strength=1.5)
#present_images(enhanced_images)
save_images(enhanced_images,enhancemented_path)

###2. Segmentation
#  using K-means algorithm
segmented_images = kmeans_segmentation(enhanced_images, 3)
save_images(segmented_images,segmented_path)

###3.  Representation
#Using gradient and then k-means to get better result
gradient_edges = gradient_edge_extraction(segmented_images)
gradient_edges = kmeans_segmentation(gradient_edges, 2)
#present_images(gradient_edges)
save_images(gradient_edges,edges_path)

#Using polygonal approximation to make my edges dilation & have closed edges
epsilon = 0.02
approximated_images = polygonal_approximation(gradient_edges, epsilon)
#present_images(approximated_images)
save_images(approximated_images,approximated_path)

###3.  Decription for each objects in image that we recognize
object_images, object_masks, compactness_list, perimeter_list = detect_objects(approximated_images, 700)
#present_images(object_images)
save_images(object_images,object_path)
#present_images(object_masks)
'''for object_mask, compactness, perimeter in zip(object_masks, compactness_list, perimeter_list):
    # Display the object mask
    plt.imshow(object_mask, cmap='gray')
    plt.title(f"Compactness: {compactness:.2f}, Perimeter: {perimeter:.2f}")
    plt.show()'''

###4.  Classifier (SVM)
#From the base we should learn it with screw and I will do this again to recognize nut and then test the SVM
dataset = []
for object_mask, compactness, perimeter in zip(object_masks, compactness_list, perimeter_list):
    data = {
        'compactness': compactness,
        'perimeter': perimeter,
        'label' : 'screw'
    }
    dataset.append(data)

#show_dataset(dataset)

###0. Reading data from address
target_shape = (250, 250)
image_list = read_images(image_path2, target_shape)
###1. Enhacement:
histograms = calculate_histograms(image_list)
save_histograms(histograms, histogram_path2)
histogram_equalized_images = histogram_equalization(image_list, histograms)
enhanced_images = unsharp_masking(histogram_equalized_images, sigma=1.0, strength=1.5)
save_images(enhanced_images,enhancemented_path2)
###2. Segmentation
segmented_images = kmeans_segmentation(enhanced_images, 3)
save_images(segmented_images,segmented_path2)
###3.  Representation
gradient_edges = gradient_edge_extraction(segmented_images)
gradient_edges = kmeans_segmentation(gradient_edges, 2)
save_images(gradient_edges,edges_path2)
epsilon = 0.02
approximated_images = polygonal_approximation(gradient_edges, epsilon)
save_images(approximated_images,approximated_path2)
object_images, object_masks, compactness_list, perimeter_list = detect_objects(approximated_images, 400)
save_images(object_images,object_path2)
###4.  Classifier (SVM)


compactness = np.array(compactness_list)
perimeter = np.array(perimeter_list)
x_train = np.column_stack((compactness, perimeter))
y_train = np.array(['screw'] * len(compactness))

dataset2 = np.array([
    [data['compactness'], data['perimeter']]  # Extract 'compactness' and 'perimeter' values from data dictionary
    for data in dataset
])

x_train = np.vstack((x_train, dataset2))  # Concatenate dataset2 to x_train
y_train = np.concatenate((y_train, ['nut'] * len(dataset2)))

svm_classifier = SVC()
svm_classifier.fit(x_train, y_train)



########################################################Test part


###0. Reading data from address
target_shape = (250, 250)
image_list = read_images(test_path, target_shape)
###1. Enhacement:
histograms = calculate_histograms(image_list)
histogram_equalized_images = histogram_equalization(image_list, histograms)
enhanced_images = unsharp_masking(histogram_equalized_images, sigma=1.0, strength=1.5)
###2. Segmentation
segmented_images = kmeans_segmentation(enhanced_images, 3)
###3.  Representation
gradient_edges = gradient_edge_extraction(segmented_images)
gradient_edges = kmeans_segmentation(gradient_edges, 2)
epsilon = 0.02
approximated_images = polygonal_approximation(gradient_edges, epsilon)
object_images, object_masks, compactness_list, perimeter_list = detect_objects(approximated_images, 400)
save_images(object_images,test_object_path)

compactness = np.array(compactness_list)
perimeter = np.array(perimeter_list)
x_test = np.column_stack((compactness, perimeter))
y_test = np.array(['screw'] * len(compactness))
y_pred = svm_classifier.predict(x_test)

screw_count = np.count_nonzero(y_pred == 'screw')
print("Number of 'screw' objects:", screw_count)