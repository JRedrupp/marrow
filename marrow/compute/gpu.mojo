"""GPU-accelerated compute kernels.

This module requires GPU compilation tools (Xcode with Metal on macOS,
CUDA toolkit on Linux).  It is NOT imported by default from
`marrow.compute` — import explicitly when GPU acceleration is desired:

    from marrow.compute.gpu import add
"""

import math
from sys import size_of

from gpu import global_idx
from gpu.host import DeviceBuffer, DeviceContext

from marrow.arrays import PrimitiveArray, FixedSizeListArray, Array
from marrow.buffers import Buffer, BitmapBuilder
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
    """GPU-accelerated add, reusing device buffers when already resident."""
    comptime native = T.native
    comptime BLOCK_SIZE = 256

    # Reuse device buffers if already on GPU, otherwise upload
    var lhs_dev: DeviceBuffer[native]
    var rhs_dev: DeviceBuffer[native]
    if left.buffer.has_device():
        lhs_dev = left.buffer.device.value().create_sub_buffer[native](
            0, length
        )
    else:
        lhs_dev = ctx.enqueue_create_buffer[native](length)
        var lhs_ptr = (
            left.buffer.ptr.bitcast[Scalar[native]]()
            + left.offset
            + left.buffer.offset
        )
        ctx.enqueue_copy(lhs_dev, lhs_ptr)

    if right.buffer.has_device():
        rhs_dev = right.buffer.device.value().create_sub_buffer[native](
            0, length
        )
    else:
        rhs_dev = ctx.enqueue_create_buffer[native](length)
        var rhs_ptr = (
            right.buffer.ptr.bitcast[Scalar[native]]()
            + right.offset
            + right.buffer.offset
        )
        ctx.enqueue_copy(rhs_dev, rhs_ptr)

    # Allocate output on device
    var out_dev = ctx.enqueue_create_buffer[native](length)

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

    # Build device-only result (no host copy — call .to_host(ctx) to read)
    var bm = BitmapBuilder.alloc(length)
    bm.unsafe_range_set(0, length, True)
    var device_bytes = length * size_of[native]()
    var buf = Buffer(UnsafePointer[UInt8, ImmutExternalOrigin](), device_bytes)
    buf.device = out_dev.create_sub_buffer[DType.uint8](0, device_bytes)
    return PrimitiveArray[T](
        length=length, offset=0, bitmap=bm^.freeze(), buffer=buf^
    )


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

    When a DeviceContext is provided, the result stays on device (GPU).
    Call `.to_host(ctx)` on the result to download to CPU memory.
    Without a context, dispatches to the SIMD-optimized CPU path.

    Args:
        left: Left operand array.
        right: Right operand array.
        ctx: Optional GPU device context for acceleration.

    Returns:
        A new PrimitiveArray where result[i] = left[i] + right[i].
        With GPU: result is device-resident; chain ops or call to_host().
        Without GPU: result is host-resident and immediately readable.
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


# ---------------------------------------------------------------------------
# GPU cosine similarity kernel
# ---------------------------------------------------------------------------


fn _cosine_similarity_gpu_kernel[
    dtype: DType
](
    vectors: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    query: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n_vectors: Int,
    dim: Int,
):
    """GPU kernel: each thread computes one vector's cosine similarity."""
    var tid = global_idx.x
    if tid < UInt(n_vectors):
        var offset = Int(tid) * dim
        var dot = Scalar[dtype](0)
        var norm_v = Scalar[dtype](0)
        var norm_q = Scalar[dtype](0)
        for j in range(dim):
            var v = vectors[offset + j]
            var q = query[j]
            dot += v * q
            norm_v += v * v
            norm_q += q * q
        var denom = math.sqrt(norm_v) * math.sqrt(norm_q)
        if denom > 0:
            result[tid] = dot / denom
        else:
            result[tid] = Scalar[dtype](0)


fn _cosine_similarity_gpu[
    T: DataType
](
    vectors: FixedSizeListArray,
    query: PrimitiveArray[T],
    n_vectors: Int,
    dim: Int,
    ctx: DeviceContext,
) raises -> PrimitiveArray[T]:
    """GPU-accelerated batch cosine similarity."""
    comptime native = T.native
    comptime BLOCK_SIZE = 256

    var n_values = n_vectors * dim

    # Upload or reuse vectors (flat child values)
    ref child = vectors.values[]
    var vec_dev: DeviceBuffer[native]
    if child.buffers[0].has_device():
        vec_dev = (
            child.buffers[0]
            .device.value()
            .create_sub_buffer[native](0, n_values)
        )
    else:
        vec_dev = ctx.enqueue_create_buffer[native](n_values)
        var vp = (
            child.buffers[0].ptr.bitcast[Scalar[native]]()
            + child.offset
            + child.buffers[0].offset
        )
        ctx.enqueue_copy(vec_dev, vp)

    # Upload or reuse query
    var query_dev: DeviceBuffer[native]
    if query.buffer.has_device():
        query_dev = query.buffer.device.value().create_sub_buffer[native](
            0, dim
        )
    else:
        query_dev = ctx.enqueue_create_buffer[native](dim)
        var qp = (
            query.buffer.ptr.bitcast[Scalar[native]]()
            + query.offset
            + query.buffer.offset
        )
        ctx.enqueue_copy(query_dev, qp)

    # Allocate output on device
    var out_dev = ctx.enqueue_create_buffer[native](n_vectors)

    # Launch kernel
    var num_blocks = math.ceildiv(n_vectors, BLOCK_SIZE)
    comptime kernel = _cosine_similarity_gpu_kernel[native]
    ctx.enqueue_function_experimental[kernel](
        vec_dev,
        query_dev,
        out_dev,
        n_vectors,
        dim,
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )

    # Build device-only result
    var bm = BitmapBuilder.alloc(n_vectors)
    bm.unsafe_range_set(0, n_vectors, True)
    var device_bytes = n_vectors * size_of[native]()
    var buf = Buffer(UnsafePointer[UInt8, ImmutExternalOrigin](), device_bytes)
    buf.device = out_dev.create_sub_buffer[DType.uint8](0, device_bytes)
    return PrimitiveArray[T](
        length=n_vectors, offset=0, bitmap=bm^.freeze(), buffer=buf^
    )
