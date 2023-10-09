import os
import torch
import asyncio
import pprint
from model import ResNet, ResidualBlock
from PIL import Image
from torchvision import transforms
from typing import List
from data_classes import Classification, Prediction
from time import time
from loguru import logger

class_to_idx = {
    'cat': 0,
    'dog': 1
}


async def predictions_to_output_class(predictions: List[torch.Tensor]) -> List[Classification]:
    classification_result_list = []
    # Sort along every row (every sample in the batch)
    sorted_probs, sorted_indices = torch.sort(predictions, dim=1, descending=True)

    # Convert to lists and move to CPU
    list_sorted_probs = sorted_probs.cpu().numpy().tolist()
    list_sorted_indices = sorted_indices.cpu().numpy().tolist()

    for icon_sorted_indices, icon_sorted_probs in zip(list_sorted_indices, list_sorted_probs):
        if icon_sorted_indices:
            try:
                classification_icon = Classification(
                    label=list(class_to_idx.keys())[icon_sorted_indices[0]],
                    certainty=icon_sorted_probs[0],
                    predictions=[Prediction(
                        label=list(class_to_idx.keys())[icon_index],
                        certainty=icon_certainty
                    ) for icon_index, icon_certainty in zip(icon_sorted_indices, icon_sorted_probs)]
                )
                classification_result_list.append(classification_icon)

            except Exception as e:
                logger.debug(f"Exception in predictions_to_output_class: {e}")
                classification_result_list.append(Classification())

    return classification_result_list


async def predict(image_list: List[Image.open], local: bool = False) -> List[torch.Tensor]:
    # Create the model and move it to the GPU if available
    device = 'cpu'
    model = ResNet(ResidualBlock, num_classes=10)
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.getcwd()) if local else '', 'models/trained_classifier.pth')))
    model = model.to(device)
    model.eval()

    # Define a transform
    preprocess = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        # Perform the transform and stack images to tensor
        input_batch = torch.stack([preprocess(image) for image in image_list])

        # Move the input and model to GPU for speed if available
        start = time()
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)

        # Tensor of shape batch_size x 1000 for ImageNet (for example)
        output_probabilities = torch.nn.functional.softmax(output, dim=1)
        logger.debug(f"prediction takes: {time() - start} seconds")

    except Exception as e:
        logger.debug(f"Prediction failed: {e}")
        output_probabilities = torch.tensor(data=[[]])

    return output_probabilities


if __name__ == '__main__':
    image_path_1 = os.path.join(os.getcwd(), 'data/cats/cat_1.png')
    image_path_2 = os.path.join(os.getcwd(), 'data/dogs/dog_1.png')
    image_paths = [image_path_1, image_path_2]
    test_image_list = [Image.open(image_path) for image_path in image_paths]

    prediction_output = asyncio.run(predict(test_image_list, local=True))
    output = asyncio.run(predictions_to_output_class(prediction_output))
    pprint.pprint(output)
