import torch
import torch.nn.functional as F
from torch import nn
from src.utils.iou import cal_iou

def get_nearest_mode_idxs(pred_trajs, gt_trajs,yaw=None,type='L1'):
        '''
            pred_trajs (batch_size, num_modes, num_timestamps, 5)
            yaw (batch_size, num_modes, 1)
            gt_trajs (batch_size, num_timestamps, 3):
            type: L1 or IoU
        '''
        if type == 'L1':
                distance = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :2]).norm(dim=-1)
                distance = (distance).sum(dim=-1)

                nearest_mode_idxs = distance.argmin(dim=-1)
                return nearest_mode_idxs
        elif type == 'IoU':
                wh = torch.tensor([[[1.7,4.0]]],device=pred_trajs.device) #(1,1,2)
                wh_batch = wh.repeat(pred_trajs.shape[0], pred_trajs.shape[1], 1) #(batch_size, num_modes, 2)
                pred_tarjs_xy = pred_trajs[:, :, 0, 0:2] #(batch_size, num_modes, 2)
                pred_trajs_bbox = torch.cat([pred_tarjs_xy,wh_batch,yaw],dim=-1) #(batch_size, num_modes, 5)
                gt_extend_mode = gt_trajs.repeat(1,pred_trajs.shape[1],1) #(batch_size, num_modes, 3)
                gt_extend_mode_bbox = torch.cat([gt_extend_mode[...,:-1], wh_batch, gt_extend_mode[...,-1:]],dim=-1) #(batch_size, num_modes, 5)
                # area, selected = oriented_box_intersection_2d(box2corners(pred_trajs_bbox),box2corners(gt_extend_mode_bbox)) # area (batch_size, num_modes)
                # nearest_mode_idxs = area.argmax(dim=-1)
                iou,_,_,_ = cal_iou(pred_trajs_bbox,gt_extend_mode_bbox) # area (batch_size, num_modes)
                nearest_mode_idxs = iou.argmax(dim=-1)
                return nearest_mode_idxs, iou

def GMMloss(pred_trajs, pred_scores, gt_trajs, yaw, nearest_mode_idxs=None,label=None,log_std_range=(-1.609, 5.0),rho_limit=0.5, upper_thr=0.7, bottom_thr=0.3 ):
        """
        Based on https://github.com/sshaoshuai/MTR/blob/master/mtr/utils/loss_utils.py
        Args:
            pred_scores (batch_size, num_modes):
            pred_trajs (batch_size, num_modes, num_timestamps, 5 or 3)
            gt_trajs (batch_size, num_timestamps, 3):
            yaw: (batch_size, num_modes)
            nearest_mode_idxs: (batch_size)
            label: (batch_size, num_modes)
        """
        assert pred_trajs.shape[-1] == 5
        assert pred_trajs.shape[-2] == 1
        batch_size = pred_trajs.shape[0]

        # if nearest_mode_idxs is None:
        #         distance = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :2]).norm(dim=-1)
        #         distance = (distance).sum(dim=-1)

        #         nearest_mode_idxs = distance.argmin(dim=-1)
        nearest_mode_bs_idxs = torch.arange(batch_size).type_as(nearest_mode_idxs)  # (batch_size, 2)
        pred_trajs = pred_trajs[...,0,:] #(bs, mode, 5)
        gt_trajs = gt_trajs[...,0,:].unsqueeze(1).repeat(1,pred_trajs.shape[1],1) #(bs,mode ,3)
        # import pdb; pdb.set_trace()
        # label assign for cls, if the IoU larger than xx then it is positive, smaller than xx then it is negative
        label=label.detach()
        # possitive
        label[label>=upper_thr] = 1
        # negtive
        label[label<bottom_thr] = 0
        label[torch.logical_and(label>=bottom_thr, label<upper_thr)] = -1

        label_mask_p = (label==1.0).reshape(-1)
        label_mask_n = (label==0.0).reshape(-1)
        
        label_mask = torch.logical_or(label_mask_p, label_mask_n)
        # if label is not None:
        #     # loss_cls = (F.cross_entropy(input=pred_scores.reshape(-1)[label_mask], target=label.detach().reshape(-1)[label_mask],reduction='none'))/label_mask.sum()
        #     loss_cls = (F.cross_entropy(input=pred_scores, target=label.detach(),reduction='none'))
        # else:
        # loss_cls = (F.cross_entropy(input=pred_scores, target=nearest_mode_idxs, reduction='none'))
        loss_cls = F.binary_cross_entropy_with_logits(pred_scores.reshape(-1)[label_mask], label.reshape(-1)[label_mask],reduction='mean')
        # update P mask for regression, the candidates need to reg is 1. nearest mode or 2. large IoU. This is to ensure GT has always has one to regress
        label_ = label.clone()
        label_[nearest_mode_bs_idxs, nearest_mode_idxs] = 1.0
        label_mask_p = (label_==1.0).reshape(-1)
        # nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idxs]  # (batch_size, num_timestamps, 5)
        gt_trajs_val = gt_trajs.reshape(-1,3)[label_mask_p] # (N_vali, 3)
        pred_trajs_val = pred_trajs.reshape(-1,5)[label_mask_p] # (N_vali, 5)
        res_trajs = gt_trajs_val[..., :2] - pred_trajs_val[..., 0:2]  # (batch_size, num_timestamps, 2)
        dx = res_trajs[..., 0]
        dy = res_trajs[..., 1]

        log_std1 = torch.clip(pred_trajs_val[..., 2], min=log_std_range[0], max=log_std_range[1])
        log_std2 = torch.clip(pred_trajs_val[..., 3], min=log_std_range[0], max=log_std_range[1])
        std1 = torch.exp(log_std1)  # (0.2m to 150m)
        std2 = torch.exp(log_std2)  # (0.2m to 150m)
        rho = torch.clip(pred_trajs_val[..., 4], min=-rho_limit, max=rho_limit)


        # -log(a^-1 * e^b) = log(a) - b
        reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho ** 2)  # (batch_size, num_timestamps)
        reg_gmm_exp = (0.5 * 1 / (1 - rho ** 2)) * (
                (dx ** 2) / (std1 ** 2) + (dy ** 2) / (std2 ** 2) - 2 * rho * dx * dy / (
                std1 * std2))  # (batch_size, num_timestamps)

        reg_loss = (reg_gmm_log_coefficient + reg_gmm_exp)
        loss_dyaw = nn.L1Loss()(yaw.reshape(-1,1)[label_mask_p],gt_trajs_val[...,-1:])

        return dict(loss_gmm=reg_loss.mean(),
                    loss_cls=loss_cls.mean(),
                    loss_yaw = loss_dyaw.mean())
    
if __name__ == '__main__':
    # debug here
    import matplotlib.pyplot as plt
    import numpy as np
    from src.utils.viz import get_color
    from matplotlib.patches import Rectangle
    from src.utils.iou import cal_iou
    fig,ax = plt.subplots()
    gt = torch.tensor([[[0.3946, 0.0096, 1.7, 4.0 ,0.0073]]])
    xy = torch.tensor(
            [[
                    [4.5337e-1, -7.2168e-1, 1.7, 4.0 ],
                    [3.373e-1,2.6512e-4, 1.7, 4.0 ],
                    [5.0439e-1,2.5978e-3, 1.7, 4.0 ],
                    [5.4834e-1,3.6125e-3, 1.7, 4.0 ],
                    [-4.9829e-1,-4.5483e-1, 1.7, 4.0 ],
                    [7.1875e-1,1.6699e-3, 1.7, 4.0 ]
            ]]
    )
    yaw = torch.tensor(
            [
                    [[-0.1718],
                     [0.0024],
                     [0.0018],
                     [0.0020],
                     [-0.1490],
                     [-0.0019]]
            ]
    )

    pred = torch.cat([xy,yaw],dim=-1)
    pred[...,1] *= 50
    gt[...,1] *= 50
    print(gt.shape,pred.shape)
    draw_pred = pred[0].cpu().numpy()
    
    for i, element in enumerate(draw_pred):
        # skip padding
        if element.sum() == 0:
            continue          
        if len(element) == 5:
          x, y, w, h, yaw = element  
        center = np.array([x,y])
        anchor_point = np.array([-h/2,-w/2])
        cos = np.cos(yaw)
        sin = np.sin(yaw)
        matrix = np.array([
          [cos,sin],
          [-sin,cos]
        ])
        anchor_point = np.dot(anchor_point,matrix) + center
        color = (get_color(int(0))).tolist()
        rect = Rectangle(anchor_point, h, w, angle=(np.rad2deg(yaw)), edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(anchor_point[0], anchor_point[1], str(int(i)), ha='center', va='center',color='r',zorder=10)
    for i, element in enumerate(gt[0].cpu().numpy()):
        # skip padding
        if element.sum() == 0:
                continue          
        if len(element) == 5:
                x, y, w, h, yaw = element  
        center = np.array([x,y])
        anchor_point = np.array([-h/2,-w/2])
        cos = np.cos(yaw)
        sin = np.sin(yaw)
        matrix = np.array([
        [cos,sin],
        [-sin,cos]
        ])
        anchor_point = np.dot(anchor_point,matrix) + center
        color = (get_color(int(2))).tolist()
        rect = Rectangle(anchor_point, h, w, angle=(np.rad2deg(yaw)), edgecolor=color, facecolor='none')
        ax.add_patch(rect)
    

    distance = (pred[..., 0:2] - gt[..., :2]).norm(dim=-1)
    nearest_mode_idxs = distance.argmin(dim=-1)
    print(nearest_mode_idxs)
    pred = pred.to('cuda')
    gt = gt.to('cuda')
    # area, selected = oriented_box_intersection_2d(box2corners(pred),box2corners(gt).repeat(1,6,1,1))
    # print(area)
    # print(area.argmax(dim=-1))
    # import pdb; pdb.set_trace()
    iou,_,_,_ = cal_iou(pred,gt.repeat(1,6,1))
    print(iou)
    print(iou.argmax(dim=-1))
#     plt.axis('equal'
    plt.axis([-3, 3, -3, 3])
    plt.savefig('./test_50.png')