import torch

from defectfirst.config import ModelConfig
from defectfirst.data.geometry import normalize_and_pad, pad_mask
from defectfirst.losses.pixels import per_image_loss
from defectfirst.models.segmentor import build_model
from defectfirst.training.checkpoint import seed_all


def test_noise_free_rectangle_can_be_memorized():
    seed_all(11)
    images = torch.zeros(8, 3, 32, 32) + 0.2
    target = torch.zeros(8, 32, 32)
    images[4:, :, 8:24, 8:24] = 0.9
    target[4:, 8:24, 8:24] = 1
    inputs = normalize_and_pad(images)
    truth, valid = pad_mask(target), pad_mask(torch.ones_like(target))
    model = build_model(ModelConfig(backbone="tiny_test", channels=32), None, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    passed = False
    for step in range(500):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)["logits"]
        score = logits[:, 1] - logits[:, 0]
        loss = per_image_loss(score, truth, valid).mean()
        loss.backward()
        optimizer.step()
        if step % 25 == 0:
            prediction = (score > 0) & (valid > 0)
            positive_iou = (
                (prediction[4:] & (truth[4:] > 0)).sum((1, 2))
                / (prediction[4:] | (truth[4:] > 0)).sum((1, 2))
            ).mean()
            normal_fpr = prediction[:4].sum() / valid[:4].sum()
            if positive_iou >= 0.95 and normal_fpr <= 0.001:
                passed = True
                break
    assert passed, f"Toy overfit gate failed: IoU={positive_iou}, normal FPR={normal_fpr}"
