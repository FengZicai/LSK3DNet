import torch
import numpy as np

def geo_loss(voxel_label, voxel_label_pre, voxel_shape, ignore_label=5, batch_size=1):
    # start = time.time()
    ratio = torch.nn.functional.log_softmax(voxel_label_pre, dim=1)  # 首先计算log_softmax，权重之后的公式
    voxel_label_pre = torch.argmax(voxel_label_pre.detach(), dim=1)  # 因为voxel_label_pre是跟踪梯度的，下面的算法中只需要值，因此要detach
    full_voxle_index = np.where(
        voxel_label.detach().cpu().numpy() != ignore_label)  # tuple 4，为了减小计算量，只计算了存在点的voxel，若计算不存在的voxel会慢很多
    full_voxle_index = list(full_voxle_index)  # 元组不可修改转变为list
    for num, index in enumerate(full_voxle_index):
        full_voxle_index[num] = index[:, np.newaxis]
    full_voxle_index = np.concatenate(full_voxle_index, axis=1)  # [full_num,4],每一行[batch,x_idx,y_idx,z_idx]
    neighbor_num = 8
    # voxel_idx = np.indices(voxel_shape)  # [3,480,360,32]
    # 找到xy平面8个邻居点
    neighbor_list = []
    zero_mat = np.zeros_like(full_voxle_index, dtype=np.int)  # voxel_idx每个点所在的voxel
    # 0维度代表batch，1维度的数据代表x轴index，（因为左移，只改变x坐标，xy平面组成二维矩阵）,
    zero_mat[:, 1] = 1
    neighbor_vox_idx_left = full_voxle_index - zero_mat
    neighbor_list.append(neighbor_vox_idx_left)
    neighbor_vox_idx_right = full_voxle_index + zero_mat
    neighbor_list.append(neighbor_vox_idx_right)

    zero_mat = np.zeros_like(full_voxle_index, dtype=np.int)
    # 1维度的数据代表y轴index（因为上移，只改变x坐标，xy平面组成二维矩阵）
    zero_mat[:, 2] = 1
    neighbor_vox_idx_up = full_voxle_index + zero_mat
    neighbor_list.append(neighbor_vox_idx_up)
    neighbor_vox_idx_down = full_voxle_index - zero_mat
    neighbor_list.append(neighbor_vox_idx_down)

    neighbor_vox_idx_left_up = neighbor_vox_idx_left + zero_mat
    neighbor_list.append(neighbor_vox_idx_left_up)
    neighbor_vox_idx_right_up = neighbor_vox_idx_right + zero_mat
    neighbor_list.append(neighbor_vox_idx_right_up)
    neighbor_vox_idx_left_down = neighbor_vox_idx_left - zero_mat
    neighbor_list.append(neighbor_vox_idx_left_down)
    neighbor_vox_idx_right_down = neighbor_vox_idx_right - zero_mat
    neighbor_list.append(neighbor_vox_idx_right_down)

    for i, neighbor in enumerate(neighbor_list):
        neighbor_list[i][:, 1] = np.clip(neighbor[:, 1], 0, voxel_shape[0] - 1)  # x出界处理
        neighbor_list[i][:, 2] = np.clip(neighbor[:, 2], 0, voxel_shape[1] - 1)  # y出界处理

    # 取出边上8个点的label和predict_label
    M_LGA = torch.zeros((full_voxle_index.shape[0],)).cuda()  # 记录平面周围8个体素与当前选中体素label不同类别的个数
    for i, neighbor in enumerate(neighbor_list):
        for batch_num, _ in enumerate(voxel_label_pre):
            evaluation = voxel_label_pre[batch_num][neighbor[:, 1], neighbor[:, 2], neighbor[:, 3]]
            ground_trues = voxel_label[batch_num][neighbor[:, 1], neighbor[:, 2], neighbor[:, 3]]
            mask = (evaluation != ground_trues).type(torch.int)  # 找出不同的体素label
            M_LGA += mask
    weight = M_LGA / neighbor_num
    w = torch.zeros([batch_size, voxel_shape[0], voxel_shape[1], voxel_shape[2]]).cuda()
    w[full_voxle_index[:, 0], full_voxle_index[:, 1], full_voxle_index[:, 2], full_voxle_index[:, 3]] = weight
    # 为了能使得w能够广播，因此在第1维度加一个维度
    loss = torch.nn.functional.nll_loss(torch.mul(ratio, w.unsqueeze(1)), voxel_label, ignore_index=ignore_label)
    # print(time.time()-start)
    return loss


def geo_loss6(voxel_label, voxel_label_pre, voxel_shape, ignore_label=5, batch_size=1):
    # start = time.time()
    ratio = torch.nn.functional.log_softmax(voxel_label_pre, dim=1)  # 首先计算log_softmax，权重之后的公式
    voxel_label_pre = torch.argmax(voxel_label_pre.detach(), dim=1)  # 因为voxel_label_pre是跟踪梯度的，下面的算法中只需要值，因此要detach
    full_voxle_index = np.where(
        voxel_label.detach().cpu().numpy() != ignore_label)  # tuple 4，为了减小计算量，只计算了存在点的voxel，若计算不存在的voxel会慢很多
    full_voxle_index = list(full_voxle_index)  # 元组不可修改转变为list
    for num, index in enumerate(full_voxle_index):
        full_voxle_index[num] = index[:, np.newaxis]
    full_voxle_index = np.concatenate(full_voxle_index, axis=1)  # [full_num,4],每一行[batch,x_idx,y_idx,z_idx]
    neighbor_num = 6
    # voxel_idx = np.indices(voxel_shape)  # [3,480,360,32]
    # 找到xyz 6个邻居点
    neighbor_list = []
    zero_mat = np.zeros_like(full_voxle_index, dtype=np.int)  # voxel_idx每个点所在的voxel
    # 0维度代表batch，1维度的数据代表x轴index，（因为左移，只改变x坐标，xy平面组成二维矩阵）,
    zero_mat[:, 1] = 1
    neighbor_vox_idx_left = full_voxle_index - zero_mat
    neighbor_list.append(neighbor_vox_idx_left)
    neighbor_vox_idx_right = full_voxle_index + zero_mat
    neighbor_list.append(neighbor_vox_idx_right)

    zero_mat = np.zeros_like(full_voxle_index, dtype=np.int)
    # 1维度的数据代表y轴index（因为上移，只改变x坐标，xy平面组成二维矩阵）
    zero_mat[:, 2] = 1
    neighbor_vox_idx_up = full_voxle_index + zero_mat
    neighbor_list.append(neighbor_vox_idx_up)
    neighbor_vox_idx_down = full_voxle_index - zero_mat
    neighbor_list.append(neighbor_vox_idx_down)

    zero_mat = np.zeros_like(full_voxle_index, dtype=np.int)
    zero_mat[:, 3] = 1
    neighbor_vox_idx_up = full_voxle_index + zero_mat
    neighbor_list.append(neighbor_vox_idx_up)
    neighbor_vox_idx_down = full_voxle_index - zero_mat
    neighbor_list.append(neighbor_vox_idx_down)

    for i, neighbor in enumerate(neighbor_list):
        neighbor_list[i][:, 1] = np.clip(neighbor[:, 1], 0, voxel_shape[0] - 1)  # x出界处理
        neighbor_list[i][:, 2] = np.clip(neighbor[:, 2], 0, voxel_shape[1] - 1)  # y出界处理
        neighbor_list[i][:, 3] = np.clip(neighbor[:, 3], 0, voxel_shape[2] - 1)  # z出界处理

    # 取出边上8个点的label和predict_label
    M_LGA = torch.zeros((full_voxle_index.shape[0],)).cuda()  # 记录平面周围8个体素与当前选中体素label不同类别的个数
    for i, neighbor in enumerate(neighbor_list):
        for batch_num, _ in enumerate(voxel_label_pre):
            evaluation = voxel_label_pre[batch_num][neighbor[:, 1], neighbor[:, 2], neighbor[:, 3]]
            ground_trues = voxel_label[batch_num][neighbor[:, 1], neighbor[:, 2], neighbor[:, 3]]
            mask = (evaluation != ground_trues).type(torch.int)  # 找出不同的体素label
            M_LGA += mask
    weight = M_LGA / neighbor_num
    w = torch.zeros([batch_size, voxel_shape[0], voxel_shape[1], voxel_shape[2]]).cuda()
    w[full_voxle_index[:, 0], full_voxle_index[:, 1], full_voxle_index[:, 2], full_voxle_index[:, 3]] = weight
    loss = torch.nn.functional.nll_loss(torch.mul(ratio, w.unsqueeze(1)), voxel_label, ignore_index=ignore_label)
    # print(time.time()-start)
    return loss