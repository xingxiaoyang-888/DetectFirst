from __future__ import annotations

import numpy as np
import torch

from defectfirst.data.geometry import letterbox, normalize_and_pad, restore_prediction


@torch.inference_mode()
def predict_image(
    model, rgb: np.ndarray, canvas: tuple[int, int], device: torch.device, amp: str = "none"
) -> np.ndarray:
    image, _, _, geometry = letterbox(rgb, canvas)
    tensor = normalize_and_pad(torch.from_numpy(image).permute(2, 0, 1)[None].float() / 255).to(
        device
    )
    was_training = model.training
    model.eval()
    try:
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=amp == "bfloat16"
        ):
            logits = model(tensor)["logits"]
        probability = torch.sigmoid(logits[0, 1] - logits[0, 0])
        return restore_prediction(probability, geometry)
    finally:
        model.train(was_training)
