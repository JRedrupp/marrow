"""GPU-accelerated compute kernels.

This module requires GPU compilation tools (Xcode with Metal on macOS,
CUDA toolkit on Linux).  It is NOT imported by default from
`marrow.compute` — import explicitly when GPU acceleration is desired:

    from marrow.compute.gpu import add
"""

import math

from gpu import global_idx
from gpu.host import DeviceContext

from marrow.arrays import PrimitiveArray, Array
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import DataType, all_numeric_dtypes, materialize
from .arithmetic import _add, _add_no_nulls
from .kernels import binary


# ---------------------------------------------------------------------------
# GPU kernel
# ---------------------------------------------------------------------------


fn _add_gpu_kernel[
    dtype: DType
](
    lhs: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rhs: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    length: Int,
):
    """GPU kernel for element-wise addition."""
    var tid = global_idx.x
    if tid < UInt(length):
        result[tid] = lhs[tid] + rhs[tid]


fn _add_gpu[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    length: Int,
    ctx: DeviceContext,
) raises -> PrimitiveArray[T]:
    """GPU-accelerated add: copies to device, runs kernel, copies back."""
    comptime native = T.native
    comptime BLOCK_SIZE = 256

    # Allocate device buffers
    var lhs_dev = ctx.enqueue_create_buffer[native](length)
    var rhs_dev = ctx.enqueue_create_buffer[native](length)
    var out_dev = ctx.enqueue_create_buffer[native](length)

    # Copy CPU → device
    var lhs_ptr = (
        left.buffer.ptr.bitcast[Scalar[native]]()
        + left.offset
        + left.buffer.offset
    )
    var rhs_ptr = (
        right.buffer.ptr.bitcast[Scalar[native]]()
        + right.offset
        + right.buffer.offset
    )
    ctx.enqueue_copy(lhs_dev, lhs_ptr)
    ctx.enqueue_copy(rhs_dev, rhs_ptr)

    # Launch kernel
    var num_blocks = math.ceildiv(length, BLOCK_SIZE)
    comptime kernel = _add_gpu_kernel[native]
    ctx.enqueue_function_experimental[kernel](
        lhs_dev,
        rhs_dev,
        out_dev,
        length,
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )

    # Copy result device → CPU
    var result = PrimitiveBuilder[T](length)
    result.bitmap.unsafe_range_set(0, length, True)
    var out_ptr = result.buffer.ptr.bitcast[Scalar[native]]()
    ctx.enqueue_copy(out_ptr, out_dev)
    ctx.synchronize()

    result.length = length
    return result^.freeze()


# ---------------------------------------------------------------------------
# Public API — mirrors arithmetic.add but with DeviceContext dispatch
# ---------------------------------------------------------------------------


fn add[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    ctx: Optional[DeviceContext] = None,
) raises -> PrimitiveArray[T]:
    """Element-wise addition with optional GPU acceleration.

    When a DeviceContext is provided, data is copied to the device,
    the kernel runs on the GPU, and the result is copied back to CPU.
    Without a context, dispatches to the SIMD-optimized CPU path.

    Args:
        left: Left operand array.
        right: Right operand array.
        ctx: Optional GPU device context for acceleration.

    Returns:
        A new PrimitiveArray where result[i] = left[i] + right[i].
        Null if either input is null at that position (CPU fallback).
    """
    if len(left) != len(right):
        raise Error(
            "add: arrays must have the same length, got {} and {}".format(
                len(left), len(right)
            )
        )

    var no_nulls = left.null_count() == 0 and right.null_count() == 0

    if no_nulls:
        if ctx:
            return _add_gpu[T](left, right, len(left), ctx.value())
        return _add_no_nulls[T](left, right, len(left))

    return binary[T, T, T, _add[T.native]](left, right)


fn add(
    left: Array, right: Array, ctx: Optional[DeviceContext] = None
) raises -> Array:
    """Runtime-typed add with optional GPU acceleration.

    Args:
        left: Left operand (runtime-typed Array).
        right: Right operand (runtime-typed Array).
        ctx: Optional GPU device context for acceleration.

    Returns:
        A new Array with the element-wise sum.
    """
    if left.dtype != right.dtype:
        raise Error(
            "add: dtype mismatch: "
            + String(left.dtype)
            + " vs "
            + String(right.dtype)
        )

    comptime for dtype in all_numeric_dtypes:
        if left.dtype == materialize[dtype]():
            return Array(
                add[dtype](
                    left.as_primitive[dtype](),
                    right.as_primitive[dtype](),
                    ctx,
                )
            )

    raise Error("add: unsupported dtype " + String(left.dtype))
