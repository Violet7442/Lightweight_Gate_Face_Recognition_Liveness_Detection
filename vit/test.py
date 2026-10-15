import torch
from thop import profile

# 导入模型类
from models import CD_MAGNet

# --------------------------
# 填入你训练时真实使用的 num_classes
# --------------------------
num_classes = 86   # 改成你训练脚本里面实际的类别数量！

model = CD_MAGNet(num_classes=num_classes)
model.eval()
model.cuda()

dummy_input = torch.randn(1, 3, 112, 112).cuda()

flops, params = profile(
    model,
    inputs=(dummy_input,),
    verbose=False
)

GMac = flops / 1e9
M_params = params / 1e6

print(f"FLOPs: {GMac:.2f} GMac")
print(f"Params: {M_params:.2f} M")
