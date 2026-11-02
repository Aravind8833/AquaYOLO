import torch
from aquayolo.evaluate import decode_predictions, collect_stats, compute_ap

def test_eval_conf_threshold():
    # Simulate model outputs for 1 image, 15 classes
    # P3: 80x80, P4: 40x40, P5: 20x20
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
    print("High conf threshold mAP:", ap_high)

    # With standard evaluation threshold 0.005 / 0.01, boxes are retained
    preds_low = decode_predictions(outputs, conf_threshold=0.005)
    assert len(preds_low[0]["boxes"]) > 0, "Candidate boxes should be retained with standard evaluation threshold"
    print(f"Low conf threshold retained {len(preds_low[0]['boxes'])} boxes")
