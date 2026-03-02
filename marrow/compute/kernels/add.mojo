"""Element-wise add kernel — CPU SIMD and GPU specializations."""

from sys import size_of, has_accelerator

from gpu.host import DeviceContext

from marrow.arrays import PrimitiveArray, Array
from marrow.buffers import MemorySpace
from marrow.dtypes import DataType
from . import binary_simd, binary_gpu, binary_array_dispatch


fn _add_simd[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
    return a + b


fn _add_gpu[T: DType](a: Scalar[T], b: Scalar[T]) -> Scalar[T]:
    return a + b


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


fn add[
    T: DataType
](left: PrimitiveArray[T], right: PrimitiveArray[T]) raises -> PrimitiveArray[
    T
]:
    """Element-wise addition of two primitive arrays of the same type.

    Uses SIMD vectorization for the data loop. Null propagation is handled
    by computing the output validity bitmap as the AND of the input bitmaps
    before the data loop (no per-element null checks).

    Args:
        left: Left operand array.
        right: Right operand array.

    Returns:
        A new PrimitiveArray where result[i] = left[i] + right[i].
        Null if either input is null at that position.
    """
    if len(left) != len(right):
        raise Error(
            "add: arrays must have the same length, got {} and {}".format(
                len(left), len(right)
            )
        )
    return binary_simd[T, _add_simd[T.native]](left, right, len(left))


fn add[
    T: DataType
](
    left: PrimitiveArray[T, MemorySpace.DEVICE],
    right: PrimitiveArray[T, MemorySpace.DEVICE],
    ctx: DeviceContext,
) raises -> PrimitiveArray[T, MemorySpace.DEVICE]:
    """Element-wise addition on device-resident arrays.

    Args:
        left: Left operand (device-resident).
        right: Right operand (device-resident).
        ctx: GPU device context.

    Returns:
        A new device-resident PrimitiveArray where result[i] = left[i] + right[i].
        Call `.to_host(ctx)` to download to CPU memory.
    """
    if len(left) != len(right):
        raise Error(
            "add: arrays must have the same length, got {} and {}".format(
                len(left), len(right)
            )
        )

    @parameter
    if has_accelerator():
        return binary_gpu[T, _add_gpu[T.native]](left, right, len(left), ctx)
    else:
        raise Error("add: no GPU accelerator available on this system")


fn add(
    left: Array[MemorySpace.CPU], right: Array[MemorySpace.CPU]
) raises -> Array[MemorySpace.CPU]:
    """Runtime-typed add: dispatches to the correct typed add.

    Args:
        left: Left operand (runtime-typed Array).
        right: Right operand (runtime-typed Array).

    Returns:
        A new Array with the element-wise sum.
    """
    return binary_array_dispatch["add", add](left, right)
