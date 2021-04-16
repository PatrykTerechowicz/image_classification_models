import torch
import matplotlib.pyplot as plt

def create_classification_figure(in_tensor: torch.Tensor, predictions_prob: torch.Tensor):
    fig = plt.figure()
    im_ax = fig.add_axes([0.1, 0.0, 0.8, 0.9])
    tensor_as_numpy = np.transpose(in_tensor.cpu().numpy(), (1, 2, 0))
    im_ax.imshow(tensor_as_numpy)
    