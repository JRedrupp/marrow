from memory import ArcPointer
from ..buffers import Buffer, Bitmap


struct ListArray(Movable, Representable, Sized, Stringable, Writable):
    var data: Array
    var capacity: Int

    fn __init__(out self, var data: Array) raises:
        if not data.dtype.is_list():
            raise Error(
                "Unexpected dtype {} instead of 'list'".format(data.dtype)
            )
        elif len(data.buffers) != 1:
            raise Error("ListArray requires exactly one buffer")
        elif len(data.children) != 1:
            raise Error("ListArray requires exactly one child array")

        self.capacity = data.length
        self.data = data^

    fn bitmap(self) -> ArcPointer[Bitmap]:
        return self.data.bitmap

    fn offsets(self) -> ArcPointer[Buffer]:
        return self.data.buffers[0]

    fn values(self) -> ArcPointer[Array]:
        return self.data.children[0]

    @staticmethod
    fn from_values(var values: Array, capacity: Int = 1) raises -> ListArray:
        """Create a ListArray wrapping the given values as its first element.

        Default capacity is at least 1 to accomodate the values.

        Args:
            values: Array to use as the first element in the ListArray.
            capacity: The capacity of the ListArray.
        """
        var list_dtype = list_(values.dtype.copy())
        var length = values.length

        var bitmap = Bitmap.alloc(capacity)
        bitmap.unsafe_set(0, True)
        var offsets = Buffer.alloc[DType.uint32](capacity + 1)
        offsets.unsafe_set[DType.uint32](0, 0)
        offsets.unsafe_set[DType.uint32](1, length)

        var data = Array(
            dtype=list_dtype^,
            length=1,
            bitmap=ArcPointer(bitmap^),
            buffers=[ArcPointer(offsets^)],
            children=[ArcPointer(values^)],
            offset=0,
        )
        return ListArray(data^)

    fn __moveinit__(out self, deinit existing: Self):
        self.data = existing.data^
        self.capacity = existing.capacity

    fn __len__(self) -> Int:
        return self.data.length

    fn is_valid(self, index: Int) -> Bool:
        return self.bitmap()[].unsafe_get(index)

    fn unsafe_append(mut self, is_valid: Bool):
        self.bitmap()[].unsafe_set(self.data.length, is_valid)
        self.offsets()[].unsafe_set[DType.uint32](
            self.data.length + 1, self.values()[].length
        )
        self.data.length += 1

    fn unsafe_get(self, index: Int, out array_data: Array) raises:
        """Access the value at a given index in the list array.

        Use an out argument to allow the caller to re-use memory while iterating over a pyarrow structure.
        """
        var start = Int(
            self.offsets()[].unsafe_get[DType.int32](self.data.offset + index)
        )
        var end = Int(
            self.offsets()[].unsafe_get[DType.int32](
                self.data.offset + index + 1
            )
        )
        ref first_child = self.data.children[0][]
        return Array(
            dtype=first_child.dtype.copy(),
            bitmap=first_child.bitmap,
            buffers=first_child.buffers.copy(),
            offset=start,
            length=end - start,
            children=first_child.children.copy(),
        )

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this ListArray to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        writer.write("ListArray(")
        writer.write("length=")
        writer.write(self.data.length)
        writer.write(")")

    fn __str__(self) -> String:
        return String.write(self)

    fn __repr__(self) -> String:
        return String.write(self)


struct StructArray(Movable, Representable, Sized, Stringable, Writable):
    var data: Array
    var fields: List[Field]
    var capacity: Int

    fn __init__(
        out self,
        var fields: List[Field],
        capacity: Int = 0,
    ):
        var bitmap = Bitmap.alloc(capacity)
        bitmap.unsafe_range_set(0, capacity, True)

        var struct_dtype = struct_(fields)

        self.capacity = capacity
        self.fields = fields^
        self.data = Array(
            dtype=struct_dtype^,
            length=0,
            bitmap=ArcPointer(bitmap^),
            buffers=[],
            children=[],
            offset=0,
        )

    fn __init__(out self, *, var data: Array):
        self.fields = data.dtype.fields.copy()
        self.capacity = data.length
        self.data = data^

    fn __moveinit__(out self, deinit existing: Self):
        self.data = existing.data^
        self.fields = existing.fields^
        self.capacity = existing.capacity

    fn __len__(self) -> Int:
        return self.data.length

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this StructArray to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        writer.write("StructArray(")
        writer.write("length=")
        writer.write(self.data.length)
        writer.write(")")

    fn _index_for_field_name(self, name: StringSlice) raises -> Int:
        for idx, ref field in enumerate(self.data.dtype.fields):
            if field.name == name:
                return idx

        raise Error("Field {} does not exist in this StructArray.".format(name))

    fn unsafe_get(
        self, name: StringSlice
    ) raises -> ref [self.data.children[0]] Array:
        """Access the field with the given name in the struct."""
        return self.data.children[self._index_for_field_name(name)][]

    fn __str__(self) -> String:
        return String.write(self)

    fn __repr__(self) -> String:
        return String.write(self)
