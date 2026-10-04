import torch

print("PyTorch:", torch.__version__)
print("ROCm/HIP:", torch.version.hip)
print("GPU disponível:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nome da GPU:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM total: {props.total_memory / 1024**3:.1f} GB")
    x = torch.randn(4096, 4096, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("Teste de cálculo na GPU: OK")