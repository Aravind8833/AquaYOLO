import torch
from aquayolo.evaluate import decode_predictions, collect_stats, compute_ap
from aquayolo.loss import aqua_loss

def test_eval_conf_threshold():
    # Simulate model outputs for 1 image, 15 classes
    bsz = 1
    nc = 15
    outputs = [
        {"obj": torch.randn(bsz, 1, 80, 80), "cls": torch.randn(bsz, nc, 80, 80), "box": torch.randn(bsz, 4, 80, 80)},
        {"obj": torch.randn(bsz, 1, 40, 40), "cls": torch.randn(bsz, nc, 40, 40), "box": torch.randn(bsz, 4, 40, 40)},
        {"obj": torch.randn(bsz, 1, 20, 20), "cls": torch.randn(bsz, nc, 20, 20), "box": torch.randn(bsz, 4, 20, 20)},
    ]
    
    # Ground truth: 1 box at class 0
    targets = [torch.tensor([[0.0, 0.5, 0.5, 0.2, 0.2]])]

    # With high threshold 0.25 on untrained / early model, everything is discarded
    preds_high = decode_predictions(outputs, conf_threshold=0.25)
    stats_high = collect_stats(preds_high, targets, iou_threshold=0.5, num_classes=nc)
    ap_high = compute_ap(stats_high[0]["scores"], stats_high[0]["matches"], stats_high[0]["num_gt"])
    assert ap_high == 0.0

    # With standard evaluation threshold 0.001, boxes are retained
    preds_low = decode_predictions(outputs, conf_threshold=0.001)
    assert len(preds_low[0]["boxes"]) > 0, "Candidate boxes should be retained with standard evaluation threshold"

def test_aqua_loss_multi_object():
    # Verify that multi-object batches calculate valid box loss without channel scrambling
    nc = 15
    outputs = [
        {"obj": torch.randn(2, 1, 80, 80), "cls": torch.randn(2, nc, 80, 80), "box": torch.randn(2, 4, 80, 80)},
        {"obj": torch.randn(2, 1, 40, 40), "cls": torch.randn(2, nc, 40, 40), "box": torch.randn(2, 4, 40, 40)},
        {"obj": torch.randn(2, 1, 20, 20), "cls": torch.randn(2, nc, 20, 20), "box": torch.randn(2, 4, 20, 20)},
    ]
    targets = [
        torch.tensor([[0.0, 0.2, 0.3, 0.1, 0.1], [1.0, 0.6, 0.7, 0.2, 0.2]]),
        torch.tensor([[2.0, 0.4, 0.5, 0.15, 0.15]]),
    ]
    total_loss, items = aqua_loss(outputs, targets, nc=nc)
    assert not torch.isnan(total_loss), "Loss should not be NaN"
    assert items["box"] > 0, "Box loss should be positive and finite"
