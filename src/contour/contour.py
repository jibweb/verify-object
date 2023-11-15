"""
Calculating the differentiable edge detection and contour based loss
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
from torch import nn
import torchvision
import kornia as K
import os
import src.config as config
from PIL import Image
from torchvision.transforms import GaussianBlur
import torch
from scipy.signal.windows import gaussian


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def imshow(input: torch.Tensor):
    out = torchvision.utils.make_grid(input, nrow=2, padding=5)
    out_np: np.ndarray = K.utils.tensor_to_image(out)
    plt.imshow(out_np)
    plt.axis('off')
    plt.show()


def imsavePNG(input: torch.Tensor, image_name_debug):
    out = torchvision.utils.make_grid(input, nrow=2, padding=5)
    out_np: np.ndarray = K.utils.tensor_to_image(out)
    plt.imshow(out_np)
    plt.axis('off')
    plt.savefig(f"../{image_name_debug}")


# https://github.com/chaddy1004/sobel-operator-pytorch/blob/master/model.py
class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

        Gx = torch.tensor([[-2.0, 0, 2.0], [-4.0, 0.0, 4.0], [-2.0, 0.0, 2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter1.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img, eps):
        x = self.filter1(img)
        x = torch.abs(x)
        x = torch.where(x >= 1e-6, x, torch.ones_like(x) * 1e-6)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)

        return x


# https://kornia-tutorials.readthedocs.io/en/latest/filtering_edges.html
def single_image_edge(img, filter="sobel", eps=1e-3): # -1
    """
    returning the tensor image containing edges
    :param img : numpy image (1, H, W) [0, 1]
    :param filter: sobel or canny
    :return

    """
    #imshow(img)
    if filter == "sobel":
        sobel_filter = Sobel()
        sobel_filter.to(device)
        x_sobel: torch.Tensor = sobel_filter(img, eps)
        return x_sobel


def order_tensor(tensor):
    rows, cols = torch.meshgrid(torch.arange(tensor.shape[0]), torch.arange(tensor.shape[1]))
    order = rows * tensor.shape[1] + cols
    unique_order = torch.unique(order)
    return unique_order.view(tensor.shape)


def img_contour_sdf(image_ref_edge):
    """
    Calculating the euclidean distance to the closest edge for each pixel in the reference image
    :param image_ref_edge: contour image reference with the shape [1, 1, h, w]
    :return distance_img: matrix with the same shape as ref_image filled with distances for each pixel
    """
    nonzero_idx = torch.nonzero(image_ref_edge, as_tuple=True)
    nonzeros = torch.stack(nonzero_idx[2:], dim=1)

    us = torch.arange(image_ref_edge.shape[2]).to(device)
    vs = torch.arange(image_ref_edge.shape[3]).to(device)
    u, v = torch.meshgrid(us, vs, indexing='ij')
    uv = torch.stack([u,v], dim=2)

    distances = torch.cdist(uv.reshape(-1, 2).float(), nonzeros.float(), p=2)
    sdf = distances.min(dim=1)[0].reshape(image_ref_edge.shape)

    return sdf / sdf.max()


def compute_sdf_image(reference_rgb, reference_mask):
    kernel = np.ones((5,5),np.uint8)
    reference_mask = (np.sum(reference_mask, 2) != 0).astype(np.float32)

    dilated_mask = cv2.dilate(reference_mask, kernel, iterations = 1)
    gray_image = cv2.cvtColor((255*reference_rgb).astype(np.uint8),
        cv2.COLOR_BGR2GRAY)
    ref_gray_tensor = torch.from_numpy(gray_image)[None, ...].to(device)
    ref_mask_tensor = torch.from_numpy(dilated_mask)[None, ...].to(device)

    image_ref = (ref_gray_tensor[:, None, ...]).float() / 255.
    image_ref_edge = single_image_edge(image_ref, filter="sobel")

    image_ref_edge = image_ref_edge * ref_mask_tensor[:, None, ...]

    # Normalizing
    image_ref_edge = (image_ref_edge - image_ref_edge.min())/(image_ref_edge.max() - image_ref_edge.min())
    image_ref_edge = torch.where(
        image_ref_edge >= 0.1,
        image_ref_edge,
        torch.zeros_like(image_ref_edge)
    )

    return img_contour_sdf(image_ref_edge) # Returning normalized vector between 0-1 (normalized by dividing by max()) # TODO: good?


def contour_loss(image_est, image_ref, image_name_debug, debug_flag):
    """
    # TODO: pass the filter name to this function and add it as a parameter of choice to the whole pipeline
    Calculating the contour loss based on the edges in estimated and reference image
    :param image_est: estimated image tensor in the shape (N, H, W) , values: [-100, 67]
    :param image_ref: referenced image tensor in the shape (N, H, W)
    """


    print("image_est", image_est)
    print("image_est.shape", image_est.shape)

    torch.autograd.set_detect_anomaly(True)
    if torch.sum(image_est) == 0:
        print(" image estimate in 0")
        image_est_norm = image_est
        # return 1e10
    else:
        image_est_norm = (image_est - torch.min(image_est) )/ (torch.max(image_est) - torch.min(image_est))
    gray_img_est = image_est_norm[:, None, ...] # (N, 1, H, W)
    gaussian = GaussianBlur(5, sigma=(0.1, 2.0))
    gray_img_est = gaussian.forward(gray_img_est)

    image_est_edge = single_image_edge(gray_img_est, filter="sobel")

    if torch.sum(image_est_edge) != 0:
        image_est_edge = torch.where(
            image_est_edge >= 0.1,
            image_est_edge,
            torch.zeros_like(image_est_edge)
        )
        image_est_edge = (image_est_edge - image_est_edge.min())/(image_est_edge.max() - image_est_edge.min())

    if torch.any(torch.isnan(image_est_edge)):
        print("nan image est  :(")
    if torch.any(torch.isinf(image_est_edge)):
        print("inf image est  :(")

    # if debug_flag:
    #     imsavePNG(image_est_edge, image_name_debug)

    # TODO check if/how we can blur image_ref_edge with a Gaussian filter AND still maintain the edge image
    d = (image_ref * image_est_edge)
    # dd = torch.sqrt(torch.where(d >= 1e-6, d, torch.ones_like(d) * 1e-6))
    # estimation_diff = torch.abs(dd - image_ref)
    estimation_diff = d

    # plt.figure(figsize=(16, 6))
    # plt.subplot(1, 3, 1)
    # plt.imshow((image_est_edge).detach().cpu().numpy()[0][0])
    # plt.axis('off')
    # plt.title("image_est_edge")
    # plt.subplot(1, 3, 2)
    # plt.imshow(image_ref.detach().cpu().numpy()[0][0])
    # plt.axis('off')
    # plt.title(f"image_ref")
    # plt.subplot(1, 3, 3)
    # plt.imshow(estimation_diff.detach().cpu().numpy()[0][0])
    # plt.axis('off')
    # plt.title(f"estimation_diff")
    # plt.tight_layout()
    # plt.show()

    if torch.any(torch.isnan(estimation_diff)):
        print("nan estimation_diff  :(")
    if torch.any(torch.isinf(estimation_diff)):
        print("inf estimation_diff  :(")

    return estimation_diff


if __name__ == '__main__':
    dataset_path = os.path.join(config.PATH_DATASET_TRACEBOT)
    scene_name = "002"
    scene_path = os.path.join(dataset_path, f"scenes/{scene_name}")
    im_id = 1
    obj_num = 1
    img_bgr: np.ndarray = cv2.imread(os.path.join(scene_path, f"rgb/{im_id:06d}.png"), cv2.IMREAD_COLOR)
    img_mask: np.ndarray = cv2.imread(os.path.join(scene_path, f"masks/003_container_{obj_num:03d}_{im_id:06d}.png"), cv2.IMREAD_COLOR)

    x_bgr: torch.Tensor = K.utils.image_to_tensor(img_bgr)  # CxHxWx
    x_bgr = x_bgr[None, ...].float() / 255.
    imshow(x_bgr)
    x_rgb: torch.Tensor = K.color.bgr_to_rgb(x_bgr)
    x_gray = K.color.rgb_to_grayscale(x_rgb)
    imshow(x_gray)

    x_mask: torch.Tensor = K.utils.image_to_tensor(img_mask)
    x_mask = x_mask[None, ...].float() / 255.
    x_mask_gray = K.color.rgb_to_grayscale(x_mask)
    imshow(x_mask_gray)

    x_gray = x_gray.type(torch.float64)
    x_mask_gray = x_mask_gray.type(torch.float64)
    x_final = torch.where(x_mask_gray > 0., x_gray, 0.)
    imshow(x_final)

    x_sobel: torch.Tensor = K.filters.sobel(x_final)
    imshow(x_sobel)

    x_laplacian: torch.Tensor = K.filters.laplacian(x_final, kernel_size=3)
    imshow(x_laplacian.clamp(0., 1.))

    x_laplacian: torch.Tensor = K.filters.canny(x_final)[0]
    imshow(x_laplacian.clamp(0., 1.))