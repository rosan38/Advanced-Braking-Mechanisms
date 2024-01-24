import os
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_transform():
    return transforms.Compose([transforms.ToTensor()])


def prepare_image(image_path, transform):
    try:
        with Image.open(image_path) as img:
            return transform(img.convert("RGB"))
    except IOError as e:
        print(f"Error opening image {image_path}: {e}")
        return None


def predict_image(model, device, image):
    image = image.to(device)
    with torch.inference_mode():
        return model([image])[0]


def draw_single_box(ax, box, score):
    x, y, xmax, ymax = box
    rect = patches.Rectangle((x, y), xmax - x, ymax - y, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y, f'Score: {score:.3f}', color='white', bbox=dict(facecolor='red', alpha=0.5))


def draw_boxes(image, predictions, output_path, threshold=0.5):
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image_np)

    for box, score in zip(predictions['boxes'], predictions['scores']):
        if score > threshold:
            draw_single_box(ax, box, score)

    plt.savefig(output_path)
    plt.close()


def main(path, test_image, save_image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=2)

    try:
        model.load_state_dict(torch.load(path, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(device).eval()
    transform = get_transform()

    test_images_path = Path(test_image)
    if not test_images_path.is_dir():
        print(f"Test images directory {test_image} does not exist.")
        return

    save_images_path = Path(save_image)
    if not save_images_path.exists():
        save_images_path.mkdir(parents=True, exist_ok=True)

    for image_file in test_images_path.iterdir():
        if image_file.is_file():
            image = prepare_image(str(image_file), transform)
            if image is not None:
                prediction = predict_image(model, device, image)
                output_file_name = f"output_{image_file.name}"
                output_path = save_images_path / output_file_name
                draw_boxes(image, prediction, str(output_path))


if __name__ == '__main__':
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    model_path = ROOT_PATH + '/pedestrian_detection_model.pth'
    test_images_dir = ROOT_PATH + '/test-images/'
    test_images_out_dir = ROOT_PATH + '/test-images-output/'
    main(model_path, test_images_dir, test_images_out_dir)
