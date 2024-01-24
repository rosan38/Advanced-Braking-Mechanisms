import os
import time

import torch
from torch.utils.data import DataLoader

# Import utilities from utils.py
from utils import PedestrianDataset, collate_fn, get_transform, get_model, load_all_data


def main():
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    dataset_path = ROOT_PATH + '/dataset'
    all_sets = ['set00', 'set01', 'set02', 'set03', 'set04', 'set05']
    train_images, train_annotations = load_all_data(dataset_path, all_sets)
    batch_size = 2
    num_epochs = 5
    learning_rate = 0.005
    num_classes = 2  # 1 class (person) + background

    # Initialize dataset and data loader
    train_dataset = PedestrianDataset(train_images, train_annotations, transform=get_transform())
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = get_model(num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Print the model structure
    print(f"Using device: {device}")
    print(f"Model structure '{model}'")

    # Set up the optimizer and learning rate scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print("Training started...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        for i, (images, targets) in enumerate(train_data_loader):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            losses.backward()
            optimizer.step()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_data_loader)}], " +
                  f"Loss: {losses.item():.4f}, Running Avg Loss: {total_loss / (i + 1):.4f}")

        epoch_duration = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{num_epochs}] train_data_loader in {epoch_duration:.2f}s, " +
              f"Average Loss: {total_loss / len(train_data_loader):.4f}")

        lr_scheduler.step()

    # Save the trained model
    torch.save(model.state_dict(), ROOT_PATH + 'pedestrian_detection_model.pth')
    print("Training completed")


if __name__ == "__main__":
    main()
