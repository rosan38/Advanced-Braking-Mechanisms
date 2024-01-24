import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

from utils import PedestrianDataset, collate_fn, get_transform, get_model, load_all_data


def load_trained_model(model_path, num_classes):
    try:
        model = get_model(num_classes)
        model.load_state_dict(torch.load(model_path))
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def calculate_iou(pred_box, gt_box):
    xA, yA = max(pred_box[0], gt_box[0]), max(pred_box[1], gt_box[1])
    xB, yB = min(pred_box[2], gt_box[2]), min(pred_box[3], gt_box[3])
    if xA >= xB or yA >= yB:
        return 0.0
    inter_area = (xB - xA) * (yB - yA)
    pred_box_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    gt_box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    return inter_area / float(pred_box_area + gt_box_area - inter_area)


def compute_ap(recall, precision):
    return np.trapz(np.array(precision), np.array(recall))


def evaluate(model, data_loader, device):
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            output = model(images)

            for i, out in enumerate(output):
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                pred_boxes = out['boxes'].cpu().numpy()

                true_label = 1 if len(gt_boxes) > 0 else 0
                predicted_label = 1 if len(pred_boxes) > 0 else 0
                true_labels.append(true_label)
                predicted_labels.append(predicted_label)

    return true_labels, predicted_labels


def main():
    print("Starting model evaluation")

    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    test_sets = ['set06', 'set07', 'set08', 'set09', 'set10']
    dataset_path = ROOT_PATH + '/dataset'
    batch_size = 2
    num_classes = 2  # 1 class (person) + background

    test_images, test_annotations = load_all_data(dataset_path, test_sets)
    test_dataset = PedestrianDataset(images=test_images, annotations=test_annotations,
                                     transform=get_transform())
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model_path = ROOT_PATH + '/pedestrian_detection_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_trained_model(model_path, num_classes).to(device)

    print(f"Model loaded and sent to device: {device}")

    true_labels, predicted_labels = evaluate(model, test_data_loader, device)

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Plotting the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Classification Report
    classification_rep = classification_report(true_labels, predicted_labels, labels=[0, 1],
                                               target_names=['Negative', 'Positive'])

    print("\nClassification Report:")
    print(classification_rep)


if __name__ == "__main__":
    main()
