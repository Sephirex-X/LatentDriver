import torch
ckpt = torch.load('your_model')['state_dict']
ckpt_new = dict()
for k,v in ckpt.items():
    if 'bert' in k:
        ckpt_new.update({k[5:]:v})
torch.save(ckpt_new,'checkpoints/pretrained_bert.pth')    