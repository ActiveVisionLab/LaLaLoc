import torch

def index_of_val(arr, value):
    return (arr == value).nonzero()


def recall_at_n(n, predictions, truths):
    results = []
    for prediction, truth in zip(predictions, truths):
        intersection = [p for p in prediction[:n] if p in truth[:n]]
        result = 1 if len(intersection) > 0 else 0
        results.append(result)
    results = torch.Tensor(results).mean()
    return results