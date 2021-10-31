import torch

src = '/mnt/750G/GIT/COVID19/src/classification_aux/rsnapneu_pretrain/timm-efficientnet-b3_384_deeplabv3plus_rsnapneu.pth_src'
dst = '/mnt/750G/GIT/COVID19/src/classification_aux/rsnapneu_pretrain/timm-efficientnet-b3_384_deeplabv3plus_rsnapneu.pth'

state_dict = torch.load(src)

state_dict_2 = {k.replace('module.', ''): v for k, v in state_dict.items()}

torch.save(state_dict_2, dst)
