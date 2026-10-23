import sys
import torch

sys.path.insert(0, "src")

from aquayolo.model import AquaYOLO


BATCH_SIZE = 4
IMAGE_SIZE = 640


def sum_tensors(obj):
    if torch.is_tensor(obj):
        return obj.float().mean()

    if isinstance(obj, dict):
        values = [sum_tensors(v) for v in obj.values()]
        return sum(values) if values else torch.tensor(0.0, device="cuda")

    if isinstance(obj, (list, tuple)):
        values = [sum_tensors(v) for v in obj]
        return sum(values) if values else torch.tensor(0.0, device="cuda")

    return torch.tensor(0.0, device="cuda")


def main():
    if not torch.cuda.is_available():
        print("CUDA is NOT available.")
        return

    device = torch.device("cuda")

    print("=" * 60)
    print("AquaYOLO Batch-4 GPU Memory Test")
    print("=" * 60)
    print(f"GPU       : {torch.cuda.get_device_name(0)}")
    print(f"Batch     : {BATCH_SIZE}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model = AquaYOLO(nc=26).to(device)
    model.train()

    print("\nRunning forward pass...")

    x = torch.randn(
        BATCH_SIZE,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
    )

    try:
        outputs = model(x)

        print("Forward pass: OK")
        print("Running backward pass...")

        loss = sum_tensors(outputs)
        loss.backward()

        torch.cuda.synchronize()

        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"Current allocated : {allocated:.2f} GB")
        print(f"Current reserved  : {reserved:.2f} GB")
        print(f"Peak allocated    : {peak:.2f} GB")
        print("Batch 4 test      : SUCCESS")
        print("=" * 60)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n" + "=" * 60)
            print("RESULT")
            print("=" * 60)
            print("Batch 4 test      : OUT OF GPU MEMORY")
            print("Use batch 2 instead.")
            print("=" * 60)
            torch.cuda.empty_cache()
        else:
            raise


if __name__ == "__main__":
    main()