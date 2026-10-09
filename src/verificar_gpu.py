import torch

# Verifica a instalação do PyTorch e o suporte à GPU via ROCm/HIP
print("PyTorch:", torch.__version__)
print("ROCm/HIP:", torch.version.hip)
print("GPU disponível:", torch.cuda.is_available())

if torch.cuda.is_available():
    # Identifica a GPU e consulta a memória de vídeo disponível
    print("Nome da GPU:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM total: {props.total_memory / 1024**3:.1f} GB")

    # Executa uma multiplicação de matrizes para validar o processamento na GPU
    x = torch.randn(4096, 4096, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("Teste de cálculo na GPU: OK")