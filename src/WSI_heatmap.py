import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
import os
import argparse
from tqdm import tqdm


def generate_heatmap(filepath):
    image = cv.imread(os.path.join(filepath, "measurements","resized_image.png"))
    
    image_data = getMetaData(filepath)
    tile_measurements = getMeasurements(filepath)

    tumor_classes = image_data["Tumor Classes"]
    # Other is not a class infered by the model, default when confidence score is below a threshold
    tumor_classes.append("Other")

    color_dict = getColors()
    tumor_colors = {
        tumor_class: list(color_dict.keys())[i]
        for i, tumor_class in enumerate(tumor_classes)
    }
    os.makedirs(os.path.join(filepath,"heatmaps"),exist_ok=True)
    class_colored = image.copy()
    for tumor_class in tumor_classes:
        measurements = np.zeros_like(image)
        for _, row in tile_measurements.iterrows():
            x = max(int(row["x"] // image_data["Downsample Level"]), 0)
            y = max(int(row["y"] // image_data["Downsample Level"]), 0)
            # print(x,y)
            if tumor_class == "Other":
                class_colored[
                    y : y + image_data["Tile Height"] // image_data["Downsample Level"],
                    x : x + image_data["Tile Width"] // image_data["Downsample Level"],
                    :,
                ] = color_dict[tumor_colors[row["Class"]]]
            else:
                measurements[
                    y : y + image_data["Tile Height"] // image_data["Downsample Level"],
                    x : x + image_data["Tile Width"] // image_data["Downsample Level"],
                    :,
                ] = row[tumor_class] * 255.0

        if tumor_class == "Other":
            continue
        heatmap = getHeatMap(measurements,blur=15,threshold=0.9)
        super_imposed_img = cv.addWeighted(heatmap, 0.25, image, 0.75, 0)
        cv.imwrite(
            os.path.join(filepath,"heatmaps",f"heatmap_{tumor_class}.png"), cv.cvtColor(super_imposed_img, cv.COLOR_BGR2RGB)
        )
    classes_blurred = cv.GaussianBlur(class_colored,(17,17),sigmaX=10)
    cv.imwrite(
        os.path.join(filepath,"heatmaps","tumor_classes.png"), cv.cvtColor(classes_blurred, cv.COLOR_BGR2RGB)
    )

def getHeatMap(image,blur,threshold=0.9):
    heatmap = cv.GaussianBlur(image,(65,65),sigmaX=blur)
    heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
    return cv.applyColorMap(heatmap, cv.COLORMAP_JET)

def getMeasurements(filepath):
    measurements_df = pd.read_csv(os.path.join(filepath,"measurements", "measurements.csv"))
    measurements_df.reset_index()
    return measurements_df


def getMetaData(filepath):
    image_dict = {}
    with open(os.path.join(filepath,"measurements", "image_data.csv"), "r") as f:
        for line in f:
            key = line.strip().split(",")[0]
            value = line.strip().split(",")[1:]
            if "Image" in key or "Downsample" in key or "Tile" in key:
                image_dict[key] = int(float(value[0]))
            elif "Pixel" in key:
                image_dict[key] = float(value[0])
            elif key == "Tumor Classes":
                image_dict[key] = value
            else:
                image_dict[key] = value[0]
    return image_dict


def getColors():
    return {
        "red": [143, 28, 49],
        "blue": [77, 102, 204],
        "green": [153, 204, 153],
        "yellow": [255, 200, 0],
    }


def display_image(image):
    # convert to integer, scale by 255 for proper shading
    plt_image = image
    plt.imshow(plt_image)
    plt.show()


def main(image_name):
    image_folder = os.path.join(".", "results", "inference")
    image_file = os.path.join(image_folder, image_name)
    generate_heatmap(image_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("im", help="Image filename")
    args = parser.parse_args()
    config = vars(args)
    image_name = config["im"]
    # image_name = "AS21057515 - 2024-05-07 19.34.42"
    main(image_name)
