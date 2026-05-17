"""
Utility functions to make predictions.
 
"""
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from typing import List, Tuple

# Step 2 & 3: Import ImageDraw from PIL along with Image
from PIL import Image, ImageDraw

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"


# Predict on a target image with a target model
def pred_and_plot_image(
    model: torch.nn.Module,
    class_names: List[str],
    image_path: str,
    image_size: Tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None,
    device: torch.device = device,
    save_annotated_path: str = None # Step 1: Add save_annotated_path as an optional argument
):
    """Predicts on a target image with a target model.

    Args:
        model (torch.nn.Module): A trained (or untrained) PyTorch model to predict on an image.
        class_names (List[str]): A list of target classes to map predictions to.
        image_path (str): Filepath to target image to predict on.
        image_size (Tuple[int, int], optional): Size to transform target image to. Defaults to (224, 224).
        transform (torchvision.transforms, optional): Transform to perform on image. Defaults to None which uses ImageNet normalization.
        device (torch.device, optional): Target device to perform prediction on. Defaults to device.
        save_annotated_path (str, optional): Path to save the copy of image with drawn predictions. Defaults to None.
    """

    # Step 2: Open image and explicitly convert to RGB
    img = Image.open(image_path).convert("RGB")

    # Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    ### Predict on image ###

    # Make sure the model is on the target device
    model.to(device)

    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(img).unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image.to(device))

    # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Construct the annotation text string
    pred_class = class_names[target_image_pred_label]
    pred_prob = target_image_pred_probs.max().item()
    text = f"Pred: {pred_class} | Prob: {pred_prob:.3f}"

    # Step 3: Draw a readable label background and prediction text with PIL.ImageDraw
    if save_annotated_path is not None:
        # Create a copy of the original image to avoid altering the source
        annotated_img = img.copy()
        draw = ImageDraw.Draw(annotated_img)
        
        # Define text position and a bounding box for the background rectangle
        # Using a simple fixed position (top-left corner) for standard presentation
        text_pos = (10, 10)
        
        # Draw a semi-transparent or solid dark background rectangle for high readability
        # text_bbox returns (left, top, right, bottom)
        text_bbox = draw.textbbox(text_pos, text)
        # Pad the background rectangle slightly
        padded_bbox = [text_bbox[0] - 5, text_bbox[1] - 5, text_bbox[2] + 5, text_bbox[3] + 5]
        
        # Draw background rectangle (dark blue/black)
        draw.rectangle(padded_bbox, fill=(0, 0, 0))
        # Draw text on top (white color)
        draw.text(text_pos, text, fill=(255, 255, 255))
        
        # Step 4: Save the annotated image to the requested path
        annotated_img.save(save_annotated_path)
        
        # Step 5: Print the saved path after successful prediction
        print(f"[INFO] Annotated prediction image saved successfully to: {save_annotated_path}")

    # Plot image via matplotlib as before
    plt.figure()
    plt.imshow(img)
    plt.title(text)
    plt.axis(False)