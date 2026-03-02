"""Array builders for constructing Arrow arrays incrementally.

`Builder` is the type-erased, ref-counted mutable core — the mutable counterpart
of `Array`.  Typed builders (`BoolBuilder`, `PrimitiveBuilder[T]`, `StringBuilder`,
`ListBuilder`, `FixedSizeListBuilder`, `StructBuilder`) are thin wrappers that
each hold an `ArcPointer[Builder]` and expose a type-safe `append` / `freeze` API
modelled after Arrow C++'s builder hierarchy.

Example
-------
    var b = PrimitiveBuilder[int64](capacity=1024)
    b.append(42)
    b.unsafe_append_null()
    var arr = b^.freeze()   # Array (int64)
"""

from memory import memcpy, ArcPointer
from sys import size_of
from .buffers import Buffer, BufferBuilder, bitmap_set
from .dtypes import *
from .arrays import Array


# ---------------------------------------------------------------------------
# Builder — type-erased mutable core
# ---------------------------------------------------------------------------


struct Builder(Movable):
    """Type-erased mutable builder — the mutable counterpart of `Array`.

    Layout mirrors `Array`:
      - `bitmap`   — null-validity bit-buffer (always directly owned)
      - `buffers`  — data buffers, each ref-counted via `ArcPointer[BufferBuilder]`
      - `children` — child builders for nested types, each an `ArcPointer[Builder]`

    Wrap in `ArcPointer[Builder]` for shared ownership.
    Call `freeze()` to consume and produce an immutable `Array`.
    """

    var dtype: DataType
    var length: Int
    var capacity: Int
    var bitmap: BufferBuilder
    var buffers: List[ArcPointer[BufferBuilder]]
    var children: List[ArcPointer[Builder]]
    var offset: Int

    fn __init__(
        out self,
        var dtype: DataType,
        length: Int,
        capacity: Int,
        var bitmap: BufferBuilder,
        var buffers: List[ArcPointer[BufferBuilder]],
        var children: List[ArcPointer[Builder]],
        offset: Int = 0,
    ):
        self.dtype = dtype^
        self.length = length
        self.capacity = capacity
        self.bitmap = bitmap^
        self.buffers = buffers^
        self.children = children^
        self.offset = offset

    fn __len__(self) -> Int:
        return self.length

    fn freeze(deinit self) -> Array:
        """Consume the builder and return an immutable `Array`.

        Calls `steal_data()` on each child `ArcPointer` — panics if any
        child still has more than one outstanding reference.
        """
        var frozen_bitmap = self.bitmap^.freeze()

        var frozen_buffers = List[Buffer]()
        while self.buffers:
            frozen_buffers.append(self.buffers.pop(0).steal_data().freeze())

        var frozen_children = List[Array]()
        while self.children:
            frozen_children.append(self.children.pop(0).steal_data().freeze())

        return Array(
            dtype=self.dtype^,
            length=self.length,
            bitmap=frozen_bitmap^,
            buffers=frozen_buffers^,
            children=frozen_children^,
            offset=self.offset,
        )


# ---------------------------------------------------------------------------
# BoolBuilder
# ---------------------------------------------------------------------------


struct BoolBuilder(Movable, Sized):
    """Builder for boolean arrays.

    buffers[0] — bit-packed boolean values
    """

    var data: ArcPointer[Builder]

    fn __init__(out self, capacity: Int = 0):
        self.data = ArcPointer(
            Builder(
                dtype=materialize[bool_](),
                length=0,
                capacity=capacity,
                bitmap=BufferBuilder.alloc_bits(capacity),
                buffers=[ArcPointer(BufferBuilder.alloc_bits(capacity))],
                children=List[ArcPointer[Builder]](),
            )
        )

    fn __len__(self) -> Int:
        return self.data[].length

    @always_inline
    fn unsafe_append(mut self, value: Bool):
        bitmap_set(self.data[].bitmap.ptr, self.data[].length, True)
        bitmap_set(self.data[].buffers[0][].ptr, self.data[].length, value)
        self.data[].length += 1

    @always_inline
    fn unsafe_append_null(mut self):
        bitmap_set(self.data[].bitmap.ptr, self.data[].length, False)
        self.data[].length += 1

    fn resize(mut self, capacity: Int):
        self.data[].bitmap.resize_bits(capacity)
        self.data[].buffers[0][].resize_bits(capacity)
        self.data[].capacity = capacity

    fn append(mut self, value: Bool):
        if self.data[].length >= self.data[].capacity:
            self.resize(max(self.data[].capacity * 2, self.data[].length + 1))
        self.unsafe_append(value)

    fn freeze(deinit self) -> Array:
        return self.data.steal_data().freeze()


# ---------------------------------------------------------------------------
# PrimitiveBuilder
# ---------------------------------------------------------------------------


struct PrimitiveBuilder[T: DataType](Movable, Sized):
    """Builder for fixed-size primitive arrays (integers, floats).

    buffers[0] — element data
    """

    comptime dtype = Self.T
    comptime scalar = Scalar[Self.T.native]

    var data: ArcPointer[Builder]

    fn __init__(out self, capacity: Int = 0):
        self.data = ArcPointer(
            Builder(
                dtype=materialize[T](),
                length=0,
                capacity=capacity,
                bitmap=BufferBuilder.alloc_bits(capacity),
                buffers=[ArcPointer(BufferBuilder.alloc[T.native](capacity))],
                children=List[ArcPointer[Builder]](),
            )
        )

    fn __len__(self) -> Int:
        return self.data[].length

    @always_inline
    fn unsafe_append(mut self, value: Self.scalar):
        bitmap_set(self.data[].bitmap.ptr, self.data[].length, True)
        self.data[].buffers[0][].unsafe_set[T.native](self.data[].length, value)
        self.data[].length += 1

    @always_inline
    fn unsafe_append_null(mut self):
        bitmap_set(self.data[].bitmap.ptr, self.data[].length, False)
        self.data[].length += 1

    fn resize(mut self, capacity: Int):
        self.data[].bitmap.resize_bits(capacity)
        self.data[].buffers[0][].resize[T.native](capacity)
        self.data[].capacity = capacity

    fn append(mut self, value: Self.scalar):
        if self.data[].length >= self.data[].capacity:
            self.resize(max(self.data[].capacity * 2, self.data[].length + 1))
        self.unsafe_append(value)

    fn extend(mut self, values: List[Self.scalar]):
        var new_len = self.data[].length + len(values)
        if new_len >= self.data[].capacity:
            self.resize(max(self.data[].capacity * 2, new_len))
        for value in values:
            self.unsafe_append(value)

    fn freeze(deinit self) -> Array:
        return self.data.steal_data().freeze()


# ---------------------------------------------------------------------------
# StringBuilder
# ---------------------------------------------------------------------------


struct StringBuilder(Movable, Sized):
    """Builder for variable-length UTF-8 string arrays.

    buffers[0] — uint32 offsets
    buffers[1] — utf-8 byte data (grown on demand)
    """

    var data: ArcPointer[Builder]

    fn __init__(out self, capacity: Int = 0):
        var offsets = BufferBuilder.alloc[DType.uint32](capacity + 1)
        offsets.unsafe_set[DType.uint32](0, 0)
        self.data = ArcPointer(
            Builder(
                dtype=materialize[string](),
                length=0,
                capacity=capacity,
                bitmap=BufferBuilder.alloc_bits(capacity),
                buffers=[
                    ArcPointer(offsets^),
                    ArcPointer(BufferBuilder.alloc[DType.uint8](capacity)),
                ],
                children=List[ArcPointer[Builder]](),
            )
        )

    fn __len__(self) -> Int:
        return self.data[].length

    fn unsafe_append(mut self, value: String):
        var index = self.data[].length
        var last_offset = self.data[].buffers[0][].ptr.bitcast[UInt32]()[index]
        var next_offset = last_offset + UInt32(len(value))
        self.data[].length += 1
        bitmap_set(self.data[].bitmap.ptr, index, True)
        self.data[].buffers[0][].unsafe_set[DType.uint32](index + 1, next_offset)
        self.data[].buffers[1][].resize[DType.uint8](next_offset)
        memcpy(
            dest=self.data[].buffers[1][].ptr + Int(last_offset),
            src=value.unsafe_ptr(),
            count=len(value),
        )

    fn unsafe_append_null(mut self):
        var index = self.data[].length
        var last_offset = self.data[].buffers[0][].ptr.bitcast[UInt32]()[index]
        self.data[].length += 1
        bitmap_set(self.data[].bitmap.ptr, index, False)
        self.data[].buffers[0][].unsafe_set[DType.uint32](index + 1, last_offset)

    fn resize(mut self, capacity: Int):
        self.data[].bitmap.resize_bits(capacity)
        self.data[].buffers[0][].resize[DType.uint32](capacity + 1)
        self.data[].capacity = capacity

    fn append(mut self, value: String):
        if self.data[].length >= self.data[].capacity:
            self.resize(max(self.data[].capacity * 2, self.data[].length + 1))
        self.unsafe_append(value)

    fn freeze(deinit self) -> Array:
        return self.data.steal_data().freeze()


# ---------------------------------------------------------------------------
# ListBuilder
# ---------------------------------------------------------------------------


struct ListBuilder(Movable, Sized):
    """Builder for variable-length list arrays.

    buffers[0]  — uint32 offsets
    children[0] — child element builder (ArcPointer[Builder])
    """

    var data: ArcPointer[Builder]

    fn __init__(out self, var child: ArcPointer[Builder], capacity: Int = 0):
        var offsets = BufferBuilder.alloc[DType.uint32](capacity + 1)
        offsets.unsafe_set[DType.uint32](0, 0)
        var child_dtype = child[].dtype.copy()
        self.data = ArcPointer(
            Builder(
                dtype=list_(child_dtype),
                length=0,
                capacity=capacity,
                bitmap=BufferBuilder.alloc_bits(capacity),
                buffers=[ArcPointer(offsets^)],
                children=[child^],
            )
        )

    fn __len__(self) -> Int:
        return self.data[].length

    fn child(self) -> ArcPointer[Builder]:
        return self.data[].children[0]

    fn unsafe_append(mut self, is_valid: Bool):
        bitmap_set(self.data[].bitmap.ptr, self.data[].length, is_valid)
        var child_length = self.data[].children[0][].length
        self.data[].buffers[0][].unsafe_set[DType.uint32](
            self.data[].length + 1, UInt32(child_length)
        )
        self.data[].length += 1

    fn unsafe_append_null(mut self):
        self.unsafe_append(False)

    fn freeze(deinit self) -> Array:
        return self.data.steal_data().freeze()


# ---------------------------------------------------------------------------
# FixedSizeListBuilder
# ---------------------------------------------------------------------------


struct FixedSizeListBuilder(Movable, Sized):
    """Builder for fixed-size list arrays.

    children[0] — child element builder (ArcPointer[Builder])
    """

    var data: ArcPointer[Builder]

    fn __init__(
        out self, var child: ArcPointer[Builder], list_size: Int, capacity: Int = 0
    ):
        var child_dtype = child[].dtype.copy()
        self.data = ArcPointer(
            Builder(
                dtype=fixed_size_list_(child_dtype, list_size),
                length=0,
                capacity=capacity,
                bitmap=BufferBuilder.alloc_bits(capacity),
                buffers=List[ArcPointer[BufferBuilder]](),
                children=[child^],
            )
        )

    fn __len__(self) -> Int:
        return self.data[].length

    fn child(self) -> ArcPointer[Builder]:
        return self.data[].children[0]

    fn unsafe_append(mut self, is_valid: Bool):
        bitmap_set(self.data[].bitmap.ptr, self.data[].length, is_valid)
        self.data[].length += 1

    fn unsafe_append_null(mut self):
        self.unsafe_append(False)

    fn freeze(deinit self) -> Array:
        return self.data.steal_data().freeze()


# ---------------------------------------------------------------------------
# StructBuilder
# ---------------------------------------------------------------------------


struct StructBuilder(Movable, Sized):
    """Builder for struct arrays.

    children[i] — field builder for field i (ArcPointer[Builder])
    """

    var data: ArcPointer[Builder]

    fn __init__(
        out self,
        var fields: List[Field],
        var field_builders: List[ArcPointer[Builder]],
        capacity: Int = 0,
    ):
        self.data = ArcPointer(
            Builder(
                dtype=struct_(fields),
                length=0,
                capacity=capacity,
                bitmap=BufferBuilder.alloc_bits(capacity),
                buffers=List[ArcPointer[BufferBuilder]](),
                children=field_builders^,
            )
        )

    fn __len__(self) -> Int:
        return self.data[].length

    fn child(self, index: Int) -> ArcPointer[Builder]:
        return self.data[].children[index]

    fn unsafe_append(mut self, is_valid: Bool):
        bitmap_set(self.data[].bitmap.ptr, self.data[].length, is_valid)
        self.data[].length += 1

    fn unsafe_append_null(mut self):
        self.unsafe_append(False)

    fn freeze(deinit self) -> Array:
        return self.data.steal_data().freeze()


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


fn array[T: DataType]() -> Array:
    """Create an empty primitive array."""
    return PrimitiveBuilder[T](0).freeze()


fn array[T: DataType](values: List[Optional[Int]]) -> Array:
    """Create a primitive array from optional ints (`None` → null)."""
    var b = PrimitiveBuilder[T](len(values))
    for value in values:
        if value:
            b.unsafe_append(Scalar[T.native](value.value()))
        else:
            b.unsafe_append_null()
    return b^.freeze()


fn array(values: List[Optional[Bool]]) -> Array:
    """Create a boolean array from optional bools (`None` → null)."""
    var b = BoolBuilder(len(values))
    for value in values:
        if value:
            b.unsafe_append(value.value())
        else:
            b.unsafe_append_null()
    return b^.freeze()


fn nulls[T: DataType](size: Int) -> Array:
    """Create a primitive array of `size` null values."""
    var b = PrimitiveBuilder[T](capacity=size)
    b.data[].length = size
    return b^.freeze()


fn arange[T: DataType](start: Int, end: Int) -> Array:
    """Create an integer array with values [start, end)."""
    comptime assert T.is_integer(), "arange() only supports integer DataTypes"
    var b = PrimitiveBuilder[T](end - start)
    for i in range(start, end):
        b.unsafe_append(Scalar[T.native](i))
    return b^.freeze()
