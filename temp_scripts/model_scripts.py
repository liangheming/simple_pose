import cv2 as cv
import torch
from torch import nn
import torch.nn.functional as F


def gauss_kernel_demo():
    kernel = cv.getGaussianKernel(11, 0)
    kernel_2d = kernel * kernel.T
    print(kernel_2d.shape)


def get_dwconv_weights():
    conv = nn.Conv2d(3, 6, 11, 1, 5, groups=3)
    print(conv.weight.shape)


class GaussTaylorKeyPointDecoder(object):
    def __init__(self, kernel_size=11, num_joints=17):
        kernel = cv.getGaussianKernel(kernel_size, 0)
        self.kernel_size = kernel_size
        self.num_joints = num_joints
        self.blur_weights = torch.from_numpy(kernel * kernel.T)[None, None, ...].repeat(num_joints, 1, 1, 1).float()

    @staticmethod
    def heat_map_to_axis(heat_map: torch.Tensor):
        """
        :param heat_map:[batch,num_joints,h,w]
        :return:
        """
        b, c, h, w = heat_map.shape
        max_val, max_idx = heat_map.view(b, c, -1).max(dim=-1, keepdim=True)
        # [batch,joints,1]
        predicts = max_idx.repeat(1, 1, 2).float()
        predicts[..., 0] = predicts[..., 0] % w
        predicts[..., 1] = (predicts[..., 1] / w).floor()
        predicts = predicts * ((max_val > 0.).repeat(1, 1, 2).float())
        return predicts, max_val

    @torch.no_grad()
    def __call__(self, heat_map: torch.Tensor, trans_inv: torch.Tensor):
        if heat_map.device != self.blur_weights.device:
            self.blur_weights = self.blur_weights.to(heat_map.device)
        coords, max_val = self.heat_map_to_axis(heat_map)
        # 2. gauss blur
        heat_map_blur = F.conv2d(heat_map, self.blur_weights, bias=None, stride=1, padding=(self.kernel_size - 1) // 2,
                                 groups=self.num_joints)
        b, c, h, w = heat_map.shape
        ori_max, _ = heat_map.view(b, c, -1).max(dim=-1)
        blur_max, _ = heat_map_blur.view(b, c, -1).max(dim=-1)
        heat_map_blur = (heat_map_blur * ori_max[..., None, None] / blur_max[..., None, None]).clamp(min=1e-10).log()

        b_idx, c_idx = torch.meshgrid(torch.arange(b), torch.arange(c))
        x_idx, y_idx = coords[..., 0].long(), coords[..., 1].long()
        b_idx, c_idx, x_idx, y_idx = b_idx.reshape(-1), c_idx.reshape(-1), x_idx.reshape(-1), y_idx.reshape(-1)
        valid_mask = (x_idx > 1) & (x_idx < w - 2) & (y_idx > 1) & (y_idx < h - 2)
        v_b_idx, v_c_idx, v_x_idx, v_y_idx = b_idx[valid_mask], c_idx[valid_mask], x_idx[valid_mask], y_idx[valid_mask]
        dx = 0.5 * (heat_map_blur[v_b_idx, v_c_idx, v_y_idx, v_x_idx + 1] -
                    heat_map_blur[v_b_idx, v_c_idx, v_y_idx, v_x_idx - 1])
        dy = 0.5 * (heat_map_blur[v_b_idx, v_c_idx, v_y_idx + 1, v_x_idx] -
                    heat_map_blur[v_b_idx, v_c_idx, v_y_idx - 1, v_x_idx])
        dxx = 0.25 * (heat_map_blur[v_b_idx, v_c_idx, v_y_idx, v_x_idx + 2] -
                      2 * heat_map_blur[v_b_idx, v_c_idx, v_y_idx, v_x_idx] +
                      heat_map_blur[v_b_idx, v_c_idx, v_y_idx, v_x_idx - 2])
        dxy = 0.25 * (heat_map_blur[v_b_idx, v_c_idx, v_y_idx + 1, v_x_idx + 1] -
                      heat_map_blur[v_b_idx, v_c_idx, v_y_idx - 1, v_x_idx + 1] -
                      heat_map_blur[v_b_idx, v_c_idx, v_y_idx + 1, v_x_idx - 1] +
                      heat_map_blur[v_b_idx, v_c_idx, v_y_idx - 1, v_x_idx - 1])
        dyy = 0.25 * (heat_map_blur[v_b_idx, v_c_idx, v_y_idx + 2, v_x_idx] -
                      2 * heat_map_blur[v_b_idx, v_c_idx, v_y_idx, v_x_idx] +
                      heat_map_blur[v_b_idx, v_c_idx, v_y_idx - 2, v_x_idx])
        derivative_valid_mask = dxx * dyy - dxy ** 2 != 0
        dxx_dxy = torch.stack([dxx, dyy], dim=-1)
        dxy_dyy = torch.stack([dxy, dyy], dim=-1)
        v_hessian = torch.stack([dxx_dxy, dxy_dyy], dim=-2)[derivative_valid_mask]
        v_derivative = torch.stack([dx, dy], dim=-1).unsqueeze(-1)[derivative_valid_mask]
        v_hessian_inv = v_hessian.inverse()
        offsets = (-v_hessian_inv @ v_derivative).transpose(1, 2).squeeze(1)

        coords = coords.view(-1, 2)
        valid_mask[valid_mask] = derivative_valid_mask
        coords[valid_mask] = coords[valid_mask] + offsets
        coords = coords.view(b, c, 2)
        xyz = torch.cat([coords, torch.ones_like(coords[..., [0]])], dim=-1)
        trans_output = torch.einsum("bcd,bad->bca", xyz, trans_inv)
        return trans_output, max_val


if __name__ == '__main__':
    input_tensor = torch.rand(size=(4, 17, 64, 48))
    decoder = GaussTaylorKeyPointDecoder()
    decoder(input_tensor, input_tensor)
