import numpy as np
import torch
from PIL import Image

from src.dataset import get_transforms
from src.emotion_predictor import preprocess


def test_preprocess_matches_training_eval_transform():
    rng = np.random.default_rng(42)
    face_bgr = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)

    # Canli cikarim on islemesi
    infer_tensor = preprocess(face_bgr)
    assert list(infer_tensor.shape) == [1, 1, 48, 48]
    assert infer_tensor.min().item() >= -1.01
    assert infer_tensor.max().item() <= 1.01

    # Egitim eval_tf ile karsilastirma (BGR -> RGB -> PIL)
    _, eval_tf = get_transforms()
    pil_img = Image.fromarray(face_bgr[:, :, ::-1])
    train_tensor = eval_tf(pil_img).unsqueeze(0)

    assert infer_tensor.shape == train_tensor.shape
    assert torch.allclose(infer_tensor, train_tensor, atol=0.05)