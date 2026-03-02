"""Element-wise subtraction kernel."""

from sys import has_accelerator

from gpu.host import DeviceContext

from marrow.arrays import PrimitiveArray, Array
from marrow.buffers import MemorySpace
from marrow.dtypes import DataType
from . import binary_elementwise, binary_gpu, binary_array_dispatch


fn _sub[T: DType](a: Scalar[T], b: Scalar[T]) -> Scalar[T]:
    return a - b


fn _sub_gpu[T: DType](a: Scalar[T], b: Scalar[T]) -> Scalar[T]:
    return a - b


fn sub[
    T: DataType
](left: PrimitiveArray[T], right: PrimitiveArray[T]) raises -> PrimitiveArray[
    T
]:
    """Element-wise subtraction of two primitive arrays of the same type.

    Null propagation is handled by computing the output validity bitmap as the
    AND of the input bitmaps before the data loop (no per-element null checks).

    Args:
        left: Left operand array.
        right: Right operand array.

    Returns:
        A new PrimitiveArray where result[i] = left[i] - right[i].
        Null if either input is null at that position.
    """
    if len(left) != len(right):
        raise Error(
            "sub: arrays must have the same length, got {} and {}".format(
                len(left), len(right)
            )
        )
    return binary_elementwise[T, T, T, _sub[T.native]](left, right, len(left))


fn sub[
    T: DataType
](
    left: PrimitiveArray[T, MemorySpace.DEVICE],
    right: PrimitiveArray[T, MemorySpace.DEVICE],
    ctx: DeviceContext,
) raises -> PrimitiveArray[T, MemorySpace.DEVICE]:
    """Element-wise subtraction on device-resident arrays.

    Args:
        left: Left operand (device-resident).
        right: Right operand (device-resident).
        ctx: GPU device context.

    Returns:
        A new device-resident PrimitiveArray where result[i] = left[i] - right[i].
        Call `.to_host(ctx)` to download to CPU memory.
    """
    if len(left) != len(right):
        raise Error(
            "sub: arrays must have the same length, got {} and {}".format(
                len(left), len(right)
            )
        )

    @parameter
    if has_accelerator():
        return binary_gpu[T, _sub_gpu[T.native]](left, right, len(left), ctx)
    else:
        raise Error("sub: no GPU accelerator available on this system")


fn sub(
    left: Array[MemorySpace.CPU], right: Array[MemorySpace.CPU]
) raises -> Array[MemorySpace.CPU]:
    """Runtime-typed sub: dispatches to the correct typed sub.

    Args:
        left: Left operand (runtime-typed Array).
        right: Right operand (runtime-typed Array).

    Returns:
        A new Array with the element-wise difference.
    """
    return binary_array_dispatch["sub", sub](left, right)
