"""Scalar (element-wise) arithmetic kernels."""

from sys import size_of
from sys.info import simd_byte_width

from marrow.arrays import PrimitiveArray, Array
from marrow.buffers import MemorySpace
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import DataType, all_numeric_dtypes, materialize
from .kernels import binary


fn _add[T: DType](a: Scalar[T], b: Scalar[T]) -> Scalar[T]:
    return a + b


fn _add_no_nulls[
    T: DataType
](
    left: PrimitiveArray[T],
    right: PrimitiveArray[T],
    length: Int,
) -> PrimitiveArray[T]:
    """SIMD-vectorized add for arrays where neither has nulls."""
    var result = PrimitiveBuilder[T](length)
    result.bitmap.unsafe_range_set(0, length, True)

    comptime native = T.native
    var lp = (
        left.buffer.ptr.bitcast[Scalar[native]]()
        + left.offset
        + left.buffer.offset
    )
    var rp = (
        right.buffer.ptr.bitcast[Scalar[native]]()
        + right.offset
        + right.buffer.offset
    )
    var op = result.buffer.ptr.bitcast[Scalar[native]]()

    comptime width = simd_byte_width() // size_of[native]()
    var i = 0
    while i + width <= length:
        (op + i).store(
            (lp + i).load[width=width]() + (rp + i).load[width=width]()
        )
        i += width
    while i < length:
        op[i] = lp[i] + rp[i]
        i += 1

    result.length = length
    return result^.freeze()


fn add[
    T: DataType
](left: PrimitiveArray[T], right: PrimitiveArray[T]) raises -> PrimitiveArray[
    T
]:
    """Element-wise addition of two primitive arrays of the same type.

    Uses SIMD vectorization when neither array has nulls.

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

    if left.null_count() == 0 and right.null_count() == 0:
        return _add_no_nulls[T](left, right, len(left))

    return binary[T, T, T, _add[T.native]](left, right)


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
                    PrimitiveArray[dtype](data=left),
                    PrimitiveArray[dtype](data=right),
                )
            )

    raise Error("add: unsupported dtype " + String(left.dtype))
