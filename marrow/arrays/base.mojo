from memory import ArcPointer
from .primitive import *
from ..buffers import Buffer, Bitmap


@fieldwise_init
struct Array(Copyable, ImplicitlyCopyable, Representable, Stringable, Writable):
    """Array is the lower level abstraction directly usable by the library consumer.

    Equivalent with https://github.com/apache/arrow/blob/7184439dea96cd285e6de00e07c5114e4919a465/cpp/src/arrow/array/data.h#L62-L84.
    """

    var dtype: DataType
    var length: Int
    var bitmap: ArcPointer[Bitmap]
    var buffers: List[ArcPointer[Buffer]]
    var children: List[ArcPointer[Array]]
    var offset: Int

    @staticmethod
    fn from_buffer[
        dtype: DataType
    ](var buffer: Buffer, length: Int) -> Array:
        """Build an Array from a buffer where all the values are not null.
        """
        var bitmap = Bitmap.alloc(length)
        bitmap.unsafe_range_set(0, length, True)
        return Array(
            dtype=materialize[dtype](),
            length=length,
            bitmap=ArcPointer(bitmap^),
            buffers=[ArcPointer(buffer^)],
            children=[],
            offset=0,
        )

    @implicit
    fn __init__[T: DataType](out self, var value: PrimitiveArray[T]):
        self = value.data

    @implicit
    fn __init__(out self, var value: StringArray):
        self = value.data

    @implicit
    fn __init__(out self, var value: ListArray):
        self = value.data

    @implicit
    fn __init__(out self, var value: StructArray):
        self = value.data

    fn __init__(out self, *, copy: Self):
        self.dtype = copy.dtype.copy()
        self.length = copy.length
        self.bitmap = copy.bitmap
        self.buffers = copy.buffers.copy()
        self.children = copy.children.copy()
        self.offset = copy.offset

    fn is_valid(self, index: Int) -> Bool:
        return self.bitmap[].unsafe_get(index + self.offset)

    fn as_primitive[T: DataType](var self) raises -> PrimitiveArray[T]:
        return PrimitiveArray[T](self^)

    fn as_int8(var self) raises -> Int8Array:
        return Int8Array(self^)

    fn as_int16(var self) raises -> Int16Array:
        return Int16Array(self^)

    fn as_int32(var self) raises -> Int32Array:
        return Int32Array(self^)

    fn as_int64(var self) raises -> Int64Array:
        return Int64Array(self^)

    fn as_uint8(var self) raises -> UInt8Array:
        return UInt8Array(self^)

    fn as_uint16(var self) raises -> UInt16Array:
        return UInt16Array(self^)

    fn as_uint32(var self) raises -> UInt32Array:
        return UInt32Array(self^)

    fn as_uint64(var self) raises -> UInt64Array:
        return UInt64Array(self^)

    fn as_float32(var self) raises -> Float32Array:
        return Float32Array(self^)

    fn as_float64(var self) raises -> Float64Array:
        return Float64Array(self^)

    fn as_string(var self) raises -> StringArray:
        return StringArray(self^)

    fn as_list(var self) raises -> ListArray:
        return ListArray(self^)

    fn _dynamic_write[W: Writer](self, index: Int, mut writer: W):
        """Write to the given stream dispatching on the dtype."""

        @parameter
        for known_type in [
            DType.bool,
            DType.int16,
            DType.int32,
            DType.int64,
            DType.int8,
            DType.float32,
            DType.float64,
            DType.uint16,
            DType.uint32,
            DType.uint64,
            DType.uint8,
        ]:
            if self.dtype.native == known_type:
                writer.write(self.buffers[0][].unsafe_get[known_type](index))
                return
        if self.dtype.is_string():
            # Should print a StringArray through the element specific write_to.
            writer.write("<str>")
            return
        writer.write("dtype=")
        writer.write(self.dtype)

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this Array to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        for i in range(self.length):
            if self.is_valid(i):
                var real_index = i + self.offset
                self._dynamic_write(real_index, writer)
            else:
                writer.write("-")
            writer.write(" ")
            if i > 10:
                break

    fn __str__(self) -> String:
        return String.write(self)

    fn __repr__(self) -> String:
        return String.write(self)

    fn append_to_array(
        deinit self: Array, mut combined: Array, start: Int
    ) -> Int:
        """Append the content self to the combined array, consumes self.

        Args:
            combined: Array to append to.
            start: Position where to append.

        Returns:
            The new start position.
        """
        combined.bitmap[].extend(self.bitmap[], start, self.length)
        combined.buffers.extend(self.buffers^)
        combined.children.extend(self.children^)
        return start + self.length
