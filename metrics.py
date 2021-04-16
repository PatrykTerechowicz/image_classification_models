import torch

def accuracy(target, net_out):
    '''
        Calculates amount of correct predictions in given batch. We suppose that net_out contains probabilities of each class.
    '''
    assert len(target) == len(net_out.shape[0]), f"Given {len(target)} target classes and {net_out.shape[0]} predictions."
    y_pred = torch.argmax(net_out, dim=1)
    return torch.sum(y_true == y_pred)

def topk_accuracy(target, net_out, k: int=3):
    '''
        Calculates amount of correct TOP-K predictions in given batch. We suppose that net_out contains probabilities of each class.
    '''
    assert len(target) == len(net_out.shape[0]), f"Given {len(target)} target classes and {net_out.shape[0]} predictions."
    assert k > 0, f"Provided incorrect k={k}, need to suffice that k>0."
    y_pred = torch.argsort(net_out, dim=1)
    correct_preds = 0
    for y, prediction in zip(target, y_pred):
        if y in prediction[:k]:
            correct_preds += 1
    return correct_preds
