import torch
from src.model import EmotionCNN, count_parameters


def test_emotion_cnn_single_and_batch_shapes():
    model = EmotionCNN(num_classes=7)
    model.eval()

    x_single = torch.randn(1, 1, 48, 48)
    x_batch = torch.randn(16, 1, 48, 48)

    with torch.no_grad():
        out_single = model(x_single)
        out_batch = model(x_batch)

    assert list(out_single.shape) == [1, 7]
    assert list(out_batch.shape) == [16, 7]
    assert count_parameters(model) > 0