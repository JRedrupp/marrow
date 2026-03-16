"""Array concatenation kernel.

Combines a list of arrays into a single array by concatenating their contents.
Matches the semantics of PyArrow's `pyarrow.concat_arrays()` and Arrow C++'s
`arrow::Concatenate()`.

Each public overload handles one concrete array type. The type-erased
`concat(List[Array])` overload dispatches to the appropriate typed overload
at runtime, mirroring the pattern used by the arithmetic and filter kernels.

Type-specific concatenation paths:
  PrimitiveArray[bool_]:       bit-packed copy via BitmapBuilder.extend.
  PrimitiveArray[numeric]:     memcpy each array's value slice (offset-aware).
  StringArray:                 adjusted uint32 offsets + concatenated value bytes.
  ListArray:                   adjusted int32 offsets + recursive child concat.
  FixedSizeListArray:          recursive child concatenation (offset * list_size).
  StructArray:                 recursive per-field child concatenation.

Validity bitmaps are concatenated in every typed overload, accounting for each
input array's logical `offset` into its backing bitmap buffer.
"""

from std.memory import memcpy

from ..arrays import (
    Array,
    PrimitiveArray,
    StringArray,
    ListArray,
    FixedSizeListArray,
    StructArray,
)
from ..buffers import Buffer, BufferBuilder
from ..bitmap import Bitmap, BitmapBuilder
from ..dtypes import *


# ---------------------------------------------------------------------------
# concat — primitive arrays (bool_ + all numeric types)
# ---------------------------------------------------------------------------


fn concat[
    T: DataType
](arrays: List[PrimitiveArray[T]]) raises -> PrimitiveArray[T]:
    """Concatenate typed primitive arrays into a single array.

    Handles both bit-packed bool_ arrays and fixed-width numeric arrays.
    Validity bitmaps and buffer contents are correctly concatenated, including
    support for arrays with non-zero offsets (slices).

    Parameters:
        T: Element DataType (bool_, int8, float32, etc.).

    Args:
        arrays: Non-empty list of primitive arrays with the same dtype.

    Raises:
        If arrays is empty.
    """
    if len(arrays) == 0:
        raise Error("concat: cannot concatenate an empty list of arrays")

    var total_length = 0
    for arr in arrays:
        total_length += arr.length

    var bm_builder = BitmapBuilder.alloc(total_length)
    var total_nulls = 0
    var dst = 0

    comptime if T == bool_:
        var bb = BitmapBuilder.alloc(total_length)
        for arr in arrays:
            var n = arr.length
            if arr.nulls == 0:
                bm_builder.set_range(dst, n, True)
            else:
                total_nulls += arr.nulls
                if arr.bitmap:
                    var bm = arr.bitmap.value()
                    bm_builder.extend(
                        Bitmap(bm._buffer, bm._offset + arr.offset, n), dst, n
                    )
                else:
                    bm_builder.set_range(dst, n, True)
            bb.extend(Bitmap(arr.buffer, arr.offset, n), dst, n)
            dst += n
        var frozen: Optional[Bitmap] = None
        if total_nulls != 0:
            frozen = bm_builder.finish(total_length)
        var bits = bb.finish(total_length)
        return PrimitiveArray[T](
            length=total_length,
            nulls=total_nulls,
            offset=0,
            bitmap=frozen,
            buffer=bits._buffer,
        )
    else:
        var vb = BufferBuilder.alloc[T.native](total_length)
        var pos = 0
        for arr in arrays:
            var n = arr.length
            if arr.nulls == 0:
                bm_builder.set_range(dst, n, True)
            else:
                total_nulls += arr.nulls
                if arr.bitmap:
                    var bm = arr.bitmap.value()
                    bm_builder.extend(
                        Bitmap(bm._buffer, bm._offset + arr.offset, n), dst, n
                    )
                else:
                    bm_builder.set_range(dst, n, True)
            memcpy(
                dest=vb.ptr.bitcast[Scalar[T.native]]() + pos,
                src=arr.buffer.unsafe_ptr[T.native](arr.offset),
                count=n,
            )
            dst += n
            pos += n
        var frozen: Optional[Bitmap] = None
        if total_nulls != 0:
            frozen = bm_builder.finish(total_length)
        return PrimitiveArray[T](
            length=total_length,
            nulls=total_nulls,
            offset=0,
            bitmap=frozen,
            buffer=vb.finish(),
        )


# ---------------------------------------------------------------------------
# concat — string arrays
# ---------------------------------------------------------------------------


fn concat(arrays: List[StringArray]) raises -> StringArray:
    """Concatenate string arrays into a single StringArray.

    Adjusts uint32 offset values and concatenates UTF-8 value bytes.
    Supports arrays with non-zero offsets (slices).

    Args:
        arrays: Non-empty list of string arrays.

    Raises:
        If arrays is empty.
    """
    if len(arrays) == 0:
        raise Error("concat: cannot concatenate an empty list of arrays")

    var total_length = 0
    for arr in arrays:
        total_length += arr.length

    # Pre-scan to compute total value bytes.
    var total_value_bytes = 0
    for arr in arrays:
        var off_start = Int(arr.offsets.unsafe_get[DType.uint32](arr.offset))
        var off_end = Int(
            arr.offsets.unsafe_get[DType.uint32](arr.offset + arr.length)
        )
        total_value_bytes += off_end - off_start

    var ob = BufferBuilder.alloc[DType.uint32](total_length + 1)
    var vb = BufferBuilder.alloc(total_value_bytes)
    var bm_builder = BitmapBuilder.alloc(total_length)
    var total_nulls = 0
    var cur_offsets_pos = 0
    var cur_value_bytes = 0

    for arr in arrays:
        var n = arr.length
        if arr.nulls == 0:
            bm_builder.set_range(cur_offsets_pos, n, True)
        else:
            total_nulls += arr.nulls
            if arr.bitmap:
                var bm = arr.bitmap.value()
                bm_builder.extend(
                    Bitmap(bm._buffer, bm._offset + arr.offset, n),
                    cur_offsets_pos,
                    n,
                )
            else:
                bm_builder.set_range(cur_offsets_pos, n, True)

        var chunk_start = Int(arr.offsets.unsafe_get[DType.uint32](arr.offset))
        var chunk_end = Int(
            arr.offsets.unsafe_get[DType.uint32](arr.offset + n)
        )
        var chunk_bytes = chunk_end - chunk_start

        for i in range(n):
            var orig = Int(arr.offsets.unsafe_get[DType.uint32](arr.offset + i))
            ob.unsafe_set[DType.uint32](
                cur_offsets_pos + i,
                UInt32(cur_value_bytes + orig - chunk_start),
            )
        cur_offsets_pos += n

        memcpy(
            dest=vb.ptr + cur_value_bytes,
            src=arr.values.ptr + chunk_start,
            count=chunk_bytes,
        )
        cur_value_bytes += chunk_bytes

    ob.unsafe_set[DType.uint32](cur_offsets_pos, UInt32(cur_value_bytes))

    var frozen: Optional[Bitmap] = None
    if total_nulls != 0:
        frozen = bm_builder.finish(total_length)
    return StringArray(
        length=total_length,
        nulls=total_nulls,
        offset=0,
        bitmap=frozen,
        offsets=ob.finish(),
        values=vb.finish(),
    )


# ---------------------------------------------------------------------------
# concat — list arrays
# ---------------------------------------------------------------------------


fn concat(arrays: List[ListArray]) raises -> ListArray:
    """Concatenate list arrays into a single ListArray.

    Adjusts int32 offset values and recursively concatenates the flat child
    values array. Supports arrays with non-zero offsets (slices).

    Args:
        arrays: Non-empty list of list arrays with the same element dtype.

    Raises:
        If arrays is empty.
    """
    if len(arrays) == 0:
        raise Error("concat: cannot concatenate an empty list of arrays")

    var total_length = 0
    for arr in arrays:
        total_length += arr.length

    var ob = BufferBuilder.alloc[DType.int32](total_length + 1)
    var bm_builder = BitmapBuilder.alloc(total_length)
    var child_chunks = List[Array]()
    var total_nulls = 0
    var cur_offsets_pos = 0
    var cur_child_pos = 0

    for arr in arrays:
        var n = arr.length
        if arr.nulls == 0:
            bm_builder.set_range(cur_offsets_pos, n, True)
        else:
            total_nulls += arr.nulls
            if arr.bitmap:
                var bm = arr.bitmap.value()
                bm_builder.extend(
                    Bitmap(bm._buffer, bm._offset + arr.offset, n),
                    cur_offsets_pos,
                    n,
                )
            else:
                bm_builder.set_range(cur_offsets_pos, n, True)

        var child_start = Int(arr.offsets.unsafe_get[DType.int32](arr.offset))
        var child_end = Int(arr.offsets.unsafe_get[DType.int32](arr.offset + n))

        var sl = Array(copy=arr.values)
        sl.offset = child_start
        sl.length = child_end - child_start
        sl.nulls = 0
        child_chunks.append(sl^)

        for i in range(n):
            var orig = Int(arr.offsets.unsafe_get[DType.int32](arr.offset + i))
            ob.unsafe_set[DType.int32](
                cur_offsets_pos + i,
                Int32(cur_child_pos + orig - child_start),
            )
        cur_offsets_pos += n
        cur_child_pos += child_end - child_start

    ob.unsafe_set[DType.int32](cur_offsets_pos, Int32(cur_child_pos))

    var combined_child = concat(child_chunks)

    var frozen: Optional[Bitmap] = None
    if total_nulls != 0:
        frozen = bm_builder.finish(total_length)
    return ListArray(
        dtype=arrays[0].dtype,
        length=total_length,
        nulls=total_nulls,
        offset=0,
        bitmap=frozen,
        offsets=ob.finish(),
        values=combined_child^,
    )


# ---------------------------------------------------------------------------
# concat — fixed-size list arrays
# ---------------------------------------------------------------------------


fn concat(arrays: List[FixedSizeListArray]) raises -> FixedSizeListArray:
    """Concatenate fixed-size list arrays into a single FixedSizeListArray.

    Recursively concatenates the flat child values array. Supports arrays
    with non-zero offsets (slices).

    Args:
        arrays: Non-empty list of fixed-size list arrays with the same dtype.

    Raises:
        If arrays is empty.
    """
    if len(arrays) == 0:
        raise Error("concat: cannot concatenate an empty list of arrays")

    var total_length = 0
    for arr in arrays:
        total_length += arr.length

    var list_size = arrays[0].dtype.size
    var bm_builder = BitmapBuilder.alloc(total_length)
    var child_chunks = List[Array]()
    var total_nulls = 0
    var dst = 0

    for arr in arrays:
        var n = arr.length
        if arr.nulls == 0:
            bm_builder.set_range(dst, n, True)
        else:
            total_nulls += arr.nulls
            if arr.bitmap:
                var bm = arr.bitmap.value()
                bm_builder.extend(
                    Bitmap(bm._buffer, bm._offset + arr.offset, n), dst, n
                )
            else:
                bm_builder.set_range(dst, n, True)
        dst += n

        var sl = Array(copy=arr.values)
        sl.offset = arr.offset * list_size
        sl.length = n * list_size
        sl.nulls = 0
        child_chunks.append(sl^)

    var combined_child = concat(child_chunks)

    var frozen: Optional[Bitmap] = None
    if total_nulls != 0:
        frozen = bm_builder.finish(total_length)
    return FixedSizeListArray(
        dtype=arrays[0].dtype,
        length=total_length,
        nulls=total_nulls,
        offset=0,
        bitmap=frozen,
        values=combined_child^,
    )


# ---------------------------------------------------------------------------
# concat — struct arrays
# ---------------------------------------------------------------------------


fn concat(arrays: List[StructArray]) raises -> StructArray:
    """Concatenate struct arrays into a single StructArray.

    Recursively concatenates each field's child array. Struct arrays always
    have offset=0, so children are directly usable.

    Args:
        arrays: Non-empty list of struct arrays with the same schema.

    Raises:
        If arrays is empty.
    """
    if len(arrays) == 0:
        raise Error("concat: cannot concatenate an empty list of arrays")

    var total_length = 0
    for arr in arrays:
        total_length += arr.length

    var bm_builder = BitmapBuilder.alloc(total_length)
    var total_nulls = 0
    var dst = 0

    for arr in arrays:
        var n = arr.length
        if arr.nulls == 0:
            bm_builder.set_range(dst, n, True)
        else:
            total_nulls += arr.nulls
            if arr.bitmap:
                var bm = arr.bitmap.value()
                bm_builder.extend(bm, dst, n)
            else:
                bm_builder.set_range(dst, n, True)
        dst += n

    var n_fields = len(arrays[0].dtype.fields)
    var combined_children = List[Array]()
    for f in range(n_fields):
        var field_chunks = List[Array]()
        for arr in arrays:
            field_chunks.append(Array(copy=arr.children[f]))
        var combined_field = concat(field_chunks)
        combined_children.append(combined_field^)

    var frozen: Optional[Bitmap] = None
    if total_nulls != 0:
        frozen = bm_builder.finish(total_length)
    return StructArray(
        dtype=arrays[0].dtype,
        length=total_length,
        nulls=total_nulls,
        bitmap=frozen,
        children=combined_children^,
    )


# ---------------------------------------------------------------------------
# concat — type-erased dispatch
# ---------------------------------------------------------------------------


fn concat(arrays: List[Array]) raises -> Array:
    """Concatenate a list of type-erased arrays into a single array.

    Dispatches to the appropriate typed overload based on the dtype of the
    first element. All arrays must have the same dtype.

    Args:
        arrays: Non-empty list of arrays with the same dtype.

    Raises:
        If arrays is empty or the dtype is unsupported.
    """
    if len(arrays) == 0:
        raise Error("concat: cannot concatenate an empty list of arrays")

    var dtype = arrays[0].dtype

    comptime for T in primitive_dtypes:
        if dtype == T:
            var typed = List[PrimitiveArray[T]](capacity=len(arrays))
            for a in arrays:
                typed.append(PrimitiveArray[T](data=a))
            return Array(concat(typed))

    if dtype.is_string():
        var typed = List[StringArray](capacity=len(arrays))
        for a in arrays:
            typed.append(StringArray(data=a))
        return Array(concat(typed))
    elif dtype.is_list():
        var typed = List[ListArray](capacity=len(arrays))
        for a in arrays:
            typed.append(ListArray(data=a))
        return Array(concat(typed))
    elif dtype.is_fixed_size_list():
        var typed = List[FixedSizeListArray](capacity=len(arrays))
        for a in arrays:
            typed.append(FixedSizeListArray(data=a))
        return Array(concat(typed))
    elif dtype.is_struct():
        var typed = List[StructArray](capacity=len(arrays))
        for a in arrays:
            typed.append(StructArray(data=a))
        return Array(concat(typed))
    else:
        raise Error("concat: unsupported dtype: ", dtype)
