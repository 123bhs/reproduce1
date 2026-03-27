import torch

print("PyTorch 版本:", torch.__version__)
print("PyTorch 构建时 CUDA 版本:", torch.version.cuda)
print("CUDA 是否可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU 数量:", torch.cuda.device_count())
    print("当前 GPU 名称:", torch.cuda.get_device_name(0))
    print("当前 GPU 计算能力:", torch.cuda.get_device_capability(0))
    # 使用正确的方法获取运行时版本
    print("CUDA 运行时 API 版本:", torch.cuda.get_runtime_version())
    
if torch.backends.cudnn.is_available():
    print("cuDNN 版本:", torch.backends.cudnn.version())
else:
    print("cuDNN 不可用")