"""Boolean and bitwise kernels."""

import math

from marrow.arrays import PrimitiveArray
from marrow.buffers import bitmap_count_ones
from marrow.dtypes import bool_ as bool_dt
from . import bitmap_and


fn count_true(array: PrimitiveArray[bool_dt]) -> Int:
    """Count True values in a bit-packed boolean array.

    Composes `bitmap_and` (for null handling) and `bitmap_count_ones`
    (SIMD popcount over bytes).

    Note: Arrow booleans are bit-packed — each buffer byte holds 8 elements.
    `reduce_simd` cannot be used here because it iterates by element count
    and treats each byte as one element, which is semantically wrong for
    bit-packed data.

    Assumes array.offset == 0.

    Args:
        array: A bit-packed boolean array.

    Returns:
        Number of True (and non-null) elements.
    """
    var n = len(array)
    if array.nulls > 0:
        var combined = bitmap_and(array.buffer, array.bitmap, n)
        return bitmap_count_ones(combined, math.ceildiv(n, 8))
    return bitmap_count_ones(array.buffer, math.ceildiv(n, 8))
