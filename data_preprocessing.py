from torchvision import datasets, transforms
import os


def preprocess_data() -> datasets.ImageFolder:
    # Normalize images with values calculated from ImageNet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Apply transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize the short side of the image to 64 keeping aspect ratio
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        normalize,  # Normalize image
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(os.getcwd(), 'data/train'), transform=transform)

    return train_dataset


if __name__ == "__main__":
    preprocess_data()
