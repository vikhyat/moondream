LATEST_REVISION = "main"


def detect_device():
    """
    Detects the appropriate device to run on, and return the device and dtype.
    """
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    elif torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    else:
        return torch.device("cpu"), torch.float32