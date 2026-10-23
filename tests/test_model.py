import torch

from aquayolo.model import AquaYOLO


def test_aquayolo_forward():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = AquaYOLO(
        nc=26
    ).to(device)

    model.eval()

    x = torch.randn(
        1,
        3,
        640,
        640,
        device=device
    )

    with torch.no_grad():
        outputs = model(x)

    assert len(outputs) == 3

    assert outputs[0]["obj"].shape == (
        1, 1, 80, 80
    )

    assert outputs[0]["cls"].shape == (
        1, 26, 80, 80
    )

    assert outputs[0]["box"].shape == (
        1, 4, 80, 80
    )

    assert outputs[1]["obj"].shape == (
        1, 1, 40, 40
    )

    assert outputs[1]["cls"].shape == (
        1, 26, 40, 40
    )

    assert outputs[1]["box"].shape == (
        1, 4, 40, 40
    )

    assert outputs[2]["obj"].shape == (
        1, 1, 20, 20
    )

    assert outputs[2]["cls"].shape == (
        1, 26, 20, 20
    )

    assert outputs[2]["box"].shape == (
        1, 4, 20, 20
    )