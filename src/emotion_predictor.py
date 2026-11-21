import json
import cv2
import torch
import torch.nn.functional as F

from src.model import EmotionCNN


def preprocess(face_bgr) -> torch.Tensor:
    """Egitimdeki eval_tf ile birebir ayni on isleme:
    BGR -> Grayscale -> 48x48 -> [0,1] -> Normalize((0.5,), (0.5,)) -> [1, 1, 48, 48]
    """
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(gray).float().div(255.0)
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0).unsqueeze(0)


class EmotionPredictor:
    def __init__(
        self,
        model_path: str = "models/final_model.pth",
        label_map_path: str = "models/label_map.json",
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        with open(label_map_path, "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)
        self.idx_to_class = {int(v): k for k, v in class_to_idx.items()}
        self.class_names = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]

        self.model = EmotionCNN(num_classes=len(self.class_names)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, face_bgr):
        tensor = preprocess(face_bgr).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        best_idx = int(probs.argmax())
        label = self.idx_to_class[best_idx]
        confidence = float(probs[best_idx])
        probs_dict = {self.idx_to_class[i]: float(probs[i]) for i in range(len(self.class_names))}
        return label, confidence, probs_dict