import pytest
import torch
import flag_gems
from flag_gems.ops.linalg_diagonal import linalg_diagonal


@pytest.mark.parametrize("shape", [(3, 3), (4, 5), (5, 4), (2, 3, 4), (3, 4, 5, 6)])
@pytest.mark.parametrize("offset", [-2, -1, 0, 1, 2])
def test_diagonal_basic(shape, offset):
    """测试基础功能：不同形状和偏移量"""
    device = flag_gems.device
    A = torch.randn(shape, device=device)
    expected = torch.linalg.diagonal(A, offset=offset)
    result = linalg_diagonal(A, offset=offset)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(3, 3), (4, 5)])
@pytest.mark.parametrize("dim1, dim2", [(0, 1), (-2, -1)])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_diagonal_with_dims(shape, dim1, dim2, offset):
    """测试指定 dim1 和 dim2"""
    device = flag_gems.device
    A = torch.randn(shape, device=device)
    expected = torch.linalg.diagonal(A, offset=offset, dim1=dim1, dim2=dim2)
    result = linalg_diagonal(A, offset=offset, dim1=dim1, dim2=dim2)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


def test_diagonal_empty():
    """测试空对角线（offset 超出范围）"""
    device = flag_gems.device
    A = torch.randn(2, 3, device=device)
    result = linalg_diagonal(A, offset=3)
    expected = torch.linalg.diagonal(A, offset=3)
    assert result.shape == expected.shape
    torch.testing.assert_close(result, expected)


def test_diagonal_2d_manual():
    """手动验证 2D 对角线值"""
    device = flag_gems.device
    A = torch.tensor([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]], device=device)
    result = linalg_diagonal(A)
    expected = torch.tensor([1, 5, 9], device=device)
    torch.testing.assert_close(result, expected)

    result = linalg_diagonal(A, offset=1)
    expected = torch.tensor([2, 6], device=device)
    torch.testing.assert_close(result, expected)

    result = linalg_diagonal(A, offset=-1)
    expected = torch.tensor([4, 8], device=device)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("shape", [(3, 3), (4, 5, 6)])
def test_diagonal_non_last_dims(shape):
    """测试在非最后两个维度上取对角线"""
    device = flag_gems.device
    A = torch.randn(shape, device=device)
    dim1, dim2 = 0, 1  # 在最前面的两个维度取对角线
    expected = torch.linalg.diagonal(A, dim1=dim1, dim2=dim2)
    result = linalg_diagonal(A, dim1=dim1, dim2=dim2)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


def test_diagonal_negative_dim():
    """测试负数 dim1/dim2"""
    device = flag_gems.device
    A = torch.randn(2, 3, 4, 5, device=device)
    # 使用 -1 和 -2 表示最后两维（默认）
    expected = torch.linalg.diagonal(A, dim1=-2, dim2=-1)
    result = linalg_diagonal(A, dim1=-2, dim2=-1)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    # 使用 -3 和 -1
    expected = torch.linalg.diagonal(A, dim1=-3, dim2=-1)
    result = linalg_diagonal(A, dim1=-3, dim2=-1)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(4, 4), (5, 6, 7)])
def test_diagonal_large_offset(shape):
    """测试 offset 等于维度大小 - 1 的情况，对角线长度为 1"""
    device = flag_gems.device
    A = torch.randn(shape, device=device)
    if len(shape) == 2:
        max_offset = min(shape) - 1
        offset = max_offset
    else:
        # 对于 3D，默认最后两维
        max_offset = min(shape[-2], shape[-1]) - 1
        offset = max_offset
    expected = torch.linalg.diagonal(A, offset=offset)
    result = linalg_diagonal(A, offset=offset)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
