import numpy as np
import torch
import torch.nn as nn
import cv2
import models
import utils
from typing import List
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

def put_predictions(image: np.ndarray, predictions: torch.Tensor, class_names: List[str]) -> np.ndarray:
    """Puts predictions text on given image. Copies image.

    Args:
        image (np.ndarray): Image to process.
        predictions (torch.Tensor): Predictions of network, should be projected to probabilities first.
        class_names (List[str]): Names of classes.

    Returns:
        np.ndarray: Image with text on int.
    """
    preds_argsorted = torch.argsort(predictions, descending=True)
    top_preds_args = preds_argsorted[:3]
    top_preds_probabilities = predictions[top_preds_args]
    top_preds_names = [class_names[n] for n in top_preds_args]
    s = [f"{name}: {prob:.2%}" for name, prob in zip(top_preds_names, top_preds_probabilities)]
    image_copy = image.copy()
    for n, text in enumerate(s):
        image_copy = cv2.putText(image_copy, text, (15, 30 + n*20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return image_copy

def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    im: np.ndarray = image.copy()
    im = np.transpose(im, [2, 0, 1])
    im = im.astype(np.float32)/255.0
    return torch.from_numpy(im)

transform = Compose([ToTensor(), Resize(112), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


if __name__ == "__main__":
    # TORCH
    model: nn.Module = models.create_model_by_name("mobilenetV2", out_classes=100)
    save = torch.load("best.pth", map_location=torch.device("cpu"))
    print(save.keys())
    model.load_state_dict(save["model"])
    model.eval()
    #CLASS NAMES
    class_names = []
    with open("class_names.txt") as file:
        class_names = [l.strip("\n\r") for l in file.readlines()]
    #OPENCV
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        outputs = model(transform(np.transpose(frame, [2, 0, 1])).unsqueeze_(0))
        output_probabilities = torch.softmax(outputs, dim=1)

        out_frame = put_predictions(frame, output_probabilities[0, :], class_names)

        # Display the resulting frame
        cv2.imshow('frame', out_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
