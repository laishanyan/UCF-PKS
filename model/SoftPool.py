from torch.autograd import Function
import torch.nn.functional as F
import torch
from torch.nn.modules.utils import _triple, _pair, _single


def soft_pool1d(x, kernel_size=2, stride=None, force_inplace=False):
    if x.is_cuda and not force_inplace:
        x = CUDA_SOFTPOOL1d.apply(x, kernel_size, stride)
        # Replace `NaN's if found
        if torch.isnan(x).any():
            return torch.nan_to_num(x)
        return x
    kernel_size = _single(kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = _single(stride)
    # Get input sizes
    _, c, d = x.size()
    # Create exponential mask (should be similar to max-like pooling)
    e_x = torch.sum(torch.exp(x),dim=1,keepdim=True)
    e_x = torch.clamp(e_x , float(0), float('inf'))
    # Apply mask to input and pool and calculate the exponential sum
    # Tensor: [b x c x d] -> [b x c x d']
    x = F.avg_pool1d(x.mul(e_x), kernel_size, stride=stride).mul_(sum(kernel_size)).div_(F.avg_pool1d(e_x, kernel_size, stride=stride).mul_(sum(kernel_size)))
    return torch.clamp(x , float(0), float('inf'))
'''
---  E N D  O F  F U N C T I O N  S O F T _ P O O L 1 D  ---
'''



'''
---  S T A R T  O F  F U N C T I O N  S O F T _ P O O L 2 D  ---
    [About]
        Function for dowsampling based on the exponenial proportion rate of pixels (soft pooling).
        If the tensor is in CUDA the custom operation is used. Alternatively, the function uses
        standard (mostly) in-place PyTorch operations for speed and reduced memory consumption.
        It is also possible to use non-inplace operations in order to improve stability.
    [Args]
        - x: PyTorch Tensor, could be in either cpu of CUDA. If in CUDA the homonym extension is used.
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - force_inplace: Bool, determines if in-place operations are to be used regardless of the CUDA
                         custom op. Mostly useful for time monitoring. Defaults to `False`.
    [Returns]
        - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
def soft_pool2d(x, kernel_size=2, stride=None, force_inplace=False):
    if x.is_cuda and not force_inplace:
        x = CUDA_SOFTPOOL2d.apply(x, kernel_size, stride)
        # Replace `NaN's if found
        if torch.isnan(x).any():
            return torch.nan_to_num(x)
        return x
    kernel_size = _pair(kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = _pair(stride)
    # Get input sizes
    _, c, h, w = x.size()
    # Create exponential mask (should be similar to max-like pooling)
    e_x = torch.sum(torch.exp(x), dim=1, keepdim=True)
    e_x = torch.clamp(e_x, float(0), float('inf'))
    # Apply mask to input and pool and calculate the exponential sum
    # Tensor: [b x c x d] -> [b x c x d']
    x = F.avg_pool2d(x.mul(e_x), kernel_size, stride=stride).mul_(sum(kernel_size)).div_(F.avg_pool2d(e_x, kernel_size, stride=stride).mul_(sum(kernel_size)))
    return torch.clamp(x, float(0), float('inf'))
'''
---  E N D  O F  F U N C T I O N  S O F T _ P O O L 2 D  ---
'''



'''
---  S T A R T  O F  F U N C T I O N  S O F T _ P O O L 3 D  ---
    [About]
        Function for dowsampling based on the exponenial proportion rate of pixels (soft pooling).
        If the tensor is in CUDA the custom operation is used. Alternatively, the function uses
        standard (mostly) in-place PyTorch operations for speed and reduced memory consumption.
        It is also possible to use non-inplace operations in order to improve stability.
    [Args]
        - x: PyTorch Tensor, could be in either cpu of CUDA. If in CUDA the homonym extension is used.
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - force_inplace: Bool, determines if in-place operations are to be used regardless of the CUDA
                         custom op. Mostly useful for time monitoring. Defaults to `False`.
    [Returns]
        - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
def soft_pool3d(x, kernel_size=2, stride=None, force_inplace=False):
    if x.is_cuda and not force_inplace:
        x = CUDA_SOFTPOOL3d.apply(x, kernel_size, stride)
        # Replace `NaN's if found
        if torch.isnan(x).any():
            return torch.nan_to_num(x)
        return x
    kernel_size = _triple(kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = _triple(stride)
    # Get input sizes
    _, c, d, h, w = x.size()
    # Create exponential mask (should be similar to max-like pooling)
    e_x = torch.sum(torch.exp(x),dim=1,keepdim=True)
    e_x = torch.clamp(e_x , float(0), float('inf'))
    # Apply mask to input and pool and calculate the exponential sum
    # Tensor: [b x c x d x h x w] -> [b x c x d' x h' x w']
    x = F.avg_pool3d(x.mul(e_x), kernel_size, stride=stride).mul_(sum(kernel_size)).div_(F.avg_pool3d(e_x, kernel_size, stride=stride).mul_(sum(kernel_size)))
    return torch.clamp(x , float(0), float('inf'))
'''
---  E N D  O F  F U N C T I O N  S O F T _ P O O L 3 D  ---
'''

class SoftPool1d(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=None, force_inplace=False):
        super(SoftPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.force_inplace = force_inplace

    def forward(self, x):
        return soft_pool1d(x, kernel_size=self.kernel_size, stride=self.stride, force_inplace=self.force_inplace)



class SoftPool2d(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=None, force_inplace=False):
        super(SoftPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.force_inplace = force_inplace

    def forward(self, x):
        return soft_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, force_inplace=self.force_inplace)



class SoftPool3d(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=None, force_inplace=False):
        super(SoftPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.force_inplace = force_inplace

    def forward(self, x):
        return soft_pool3d(x, kernel_size=self.kernel_size, stride=self.stride, force_inplace=self.force_inplace)