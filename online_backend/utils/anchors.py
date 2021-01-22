import numpy as np
INPUT_SIZE = (448, 448)

# =============================================================================
# _default_anchors_setting = (
#     dict(layer='p3', stride=32, size=48, scale=[2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
#     dict(layer='p4', stride=64, size=96, scale=[2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
# # =============================================================================
# #     dict(layer='p5', stride=128, size=192, scale=[1, 2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
# # =============================================================================
# )
# =============================================================================
_default_anchors_setting_small = (
    dict(layer='p3', stride=32, size=48, scale=[1], aspect_ratio=[1]),
    dict(layer='p4', stride=64, size=96, scale=[1], aspect_ratio=[1]),
)
_default_anchors_setting_large = (
    dict(layer='p5', stride=128, size=192, scale=[1], aspect_ratio=[1]),
)

def generate_default_anchor_maps(anchors_setting=None, input_shape=INPUT_SIZE, 
                                 setting='small'):
    """
    generate default anchor

    :param anchors_setting: all informations of anchors
    :param input_shape: shape of input images, e.g. (h, w)
    :return: center_anchors: # anchors * 4 (oy, ox, h, w)
             edge_anchors: # anchors * 4 (y0, x0, y1, x1)
             anchor_area: # anchors * 1 (area)
    """
    if anchors_setting is None:
        if setting == 'small':
            anchors_setting = _default_anchors_setting_small
        else:
            anchors_setting = _default_anchors_setting_large

    center_anchors = np.zeros((0, 4), dtype=np.float32)
    edge_anchors = np.zeros((0, 4), dtype=np.float32)
    anchor_areas = np.zeros((0,), dtype=np.float32)
    input_shape = np.array(input_shape, dtype=int)

    for anchor_info in anchors_setting:

        stride = anchor_info['stride']
        size = anchor_info['size']
        scales = anchor_info['scale']
        aspect_ratios = anchor_info['aspect_ratio']

        output_map_shape = np.ceil(input_shape.astype(np.float32) / stride)
        output_map_shape = output_map_shape.astype(np.int)
        output_shape = tuple(output_map_shape) + (4,)
        ostart = stride / 2.
        oy = np.arange(ostart, ostart + stride * output_shape[0], stride)
        oy = oy.reshape(output_shape[0], 1)
        ox = np.arange(ostart, ostart + stride * output_shape[1], stride)
        ox = ox.reshape(1, output_shape[1])
        center_anchor_map_template = np.zeros(output_shape, dtype=np.float32)
        center_anchor_map_template[:, :, 0] = oy
        center_anchor_map_template[:, :, 1] = ox
        for scale in scales:
            for aspect_ratio in aspect_ratios:
                center_anchor_map = center_anchor_map_template.copy()
                center_anchor_map[:, :, 2] = size * scale / float(aspect_ratio) ** 0.5
                center_anchor_map[:, :, 3] = size * scale * float(aspect_ratio) ** 0.5

                edge_anchor_map = np.concatenate((center_anchor_map[..., :2] - center_anchor_map[..., 2:4] / 2.,
                                                  center_anchor_map[..., :2] + center_anchor_map[..., 2:4] / 2.),
                                                 axis=-1)
                anchor_area_map = center_anchor_map[..., 2] * center_anchor_map[..., 3]
                center_anchors = np.concatenate((center_anchors, center_anchor_map.reshape(-1, 4)))
                edge_anchors = np.concatenate((edge_anchors, edge_anchor_map.reshape(-1, 4)))
                anchor_areas = np.concatenate((anchor_areas, anchor_area_map.reshape(-1)))

    return center_anchors, edge_anchors, anchor_areas


def hard_nms(cdds, topn=10, iou_thresh=0.25):
    if not (type(cdds).__module__ == 'numpy' and len(cdds.shape) == 2 and cdds.shape[1] >= 5):
        raise TypeError('edge_box_map should be N * 5+ ndarray')

    cdds = cdds.copy()
    indices = np.argsort(cdds[:, 0])
    cdds = cdds[indices]
    cdd_results = []

    res = cdds

    while res.any():
        cdd = res[-1]
        cdd_results.append(cdd)
        if len(cdd_results) == topn:
            return np.array(cdd_results)
        res = res[:-1]

        start_max = np.maximum(res[:, 1:3], cdd[1:3])
        end_min = np.minimum(res[:, 3:5], cdd[3:5])
        lengths = end_min - start_max
        intersec_map = lengths[:, 0] * lengths[:, 1]
        intersec_map[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0
        iou_map_cur = intersec_map / ((res[:, 3] - res[:, 1]) * (res[:, 4] - res[:, 2]) + (cdd[3] - cdd[1]) * (
            cdd[4] - cdd[2]) - intersec_map)
        res = res[iou_map_cur <= iou_thresh]

    return np.array(cdd_results)

# =============================================================================
# def cdds2attention_map(top_n_cdds):
#     import torch
#     attention_map_lst = []
#     for i in range(len(top_n_cdds)):
#         attention_map = np.zeros(INPUT_SIZE)
#         SUM = np.sum(top_n_cdds[i][:, 0])
#         for j in range(len(top_n_cdds[i])):
#             [y0, x0, y1, x1] = top_n_cdds[i][j, 1:5].astype(np.int)
#             y0 = y0 - 224
#             x0 = x0 - 224
#             y1 = y1 - 224
#             x1 = x1 - 224
#             attention_map[y0:y1, x0:x1] += (top_n_cdds[i][j][0] / SUM)
#         attention_map_lst.append(attention_map)
#     return torch.Tensor(np.array(attention_map_lst)).cuda()
# =============================================================================
def cdds2attention_map(top_n_cdds):
    attention_box_lst = []
    attention_map_lst = []
    c = 600/448
    for i in range(len(top_n_cdds)):
        att_map = np.zeros((448, 448))
        MIN_y = 448
        MAX_y = 0
        MIN_x = 448
        MAX_x = 0
        SUM = top_n_cdds[i][:, 0].sum()
        for j in range(3):
            [y0, x0, y1, x1] = top_n_cdds[i][j, 1:5].astype(np.int)
            y0, x0, y1, x1 = get_xy(y0, x0, y1, x1)
            y0 = y0 - 224
            x0 = x0 - 224
            y1 = y1 - 224
            x1 = x1 - 224
            if MIN_y > y0:
                MIN_y = y0
            if MAX_y < y1:
                MAX_y = y1
            if MIN_x > x0:
                MIN_x = x0
            if MAX_x < x1:
                MAX_x = x1
            att_map[y0:y1, x0:x1] += top_n_cdds[i][j][0] / SUM
        attention_box_lst.append(
                [np.int(MIN_y*c), np.int(MIN_x*c), 
                 np.int(MAX_y*c), np.int(MAX_x*c)])
        attention_map_lst.append(att_map)
    return attention_box_lst, attention_map_lst

def get_xy(y0, x0, y1, x1, size=448):
    pad_size = size//2
    
    y0 = np.max([y0, pad_size])
    y0 = np.min([y0, size+pad_size])
    
    x0 = np.max([x0, pad_size])
    x0 = np.min([x0, size+pad_size])
    
    y1 = np.max([y1, pad_size])
    y1 = np.min([y1, size+pad_size])
    
    x1 = np.max([x1, pad_size])
    x1 = np.min([x1, size+pad_size])
    
    return y0, x0, y1, x1
if __name__ == '__main__':
    a = hard_nms(np.array([
        [0.4, 1, 10, 12, 20],
        [0.5, 1, 11, 11, 20],
        [0.55, 20, 30, 40, 50]
    ]), topn=100, iou_thresh=0.4)
    print(a)
