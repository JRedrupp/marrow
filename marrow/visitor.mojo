from .arrays import *
from .buffers import MemorySpace
from .dtypes import *


trait ArrayVisitor:
    """Trait for type-dispatched array operations.

    Implement this trait and call `visitor.visit(array)` to receive a
    concretely-typed array matching the runtime dtype of the Array.

    All typed `visit` overloads have default no-op bodies, so implementors
    only need to override the array kinds they care about. `visit(Array)`
    dispatches to the typed overloads by default. `visit(ChunkedArray)`
    dispatches to each chunk by default.

    Typed arrays passed to visitor methods share the underlying buffer memory
    with the original Array (ArcPointer semantics). Visitor methods are
    `raises` to allow implementations that perform I/O or recursive dispatch
    into nested arrays.
    """

    fn visit[T: DataType, space: MemorySpace = MemorySpace.CPU](
        mut self, array: PrimitiveArray[T, space]
    ) raises:
        pass

    fn visit[space: MemorySpace = MemorySpace.CPU](
        mut self, array: StringArray[space]
    ) raises:
        pass

    fn visit[space: MemorySpace = MemorySpace.CPU](
        mut self, array: ListArray[space]
    ) raises:
        pass

    fn visit[space: MemorySpace = MemorySpace.CPU](
        mut self, array: FixedSizeListArray[space]
    ) raises:
        pass

    fn visit[space: MemorySpace = MemorySpace.CPU](
        mut self, array: StructArray[space]
    ) raises:
        pass

    fn visit[space: MemorySpace = MemorySpace.CPU](
        mut self, array: ChunkedArray[space]
    ) raises:
        for chunk in array.chunks:
            self.visit[space](chunk)

    fn visit[space: MemorySpace = MemorySpace.CPU](
        mut self, array: Array[space]
    ) raises:
        """Dispatch to the typed overload matching the runtime dtype."""

        comptime for dtype in [
            bool_,
            int8,
            int16,
            int32,
            int64,
            uint8,
            uint16,
            uint32,
            uint64,
            float16,
            float32,
            float64,
        ]:
            if array.dtype == materialize[dtype]():
                self.visit[dtype, space](
                    PrimitiveArray[dtype, space](data=array)
                )
                return

        if array.dtype.is_string():
            self.visit[space](StringArray[space](data=array))
        elif array.dtype.is_list():
            self.visit[space](ListArray[space](data=array))
        elif array.dtype.is_fixed_size_list():
            self.visit[space](FixedSizeListArray[space](data=array))
        elif array.dtype.is_struct():
            self.visit[space](StructArray[space](data=array))
        else:
            raise Error("visit: unsupported dtype " + String(array.dtype))
