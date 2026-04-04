import torch
try:
    # 尝试加载
    path = "Data_files/MIT/ALL_feats_mit.pt"
    state_dict = torch.load(path, map_location="cpu")
    print("文件完整，可以正常加载。")
except Exception as e:
    print(f"文件损坏：{e}")