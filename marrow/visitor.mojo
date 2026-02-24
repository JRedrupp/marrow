from .arrays import *
from .dtypes import *


trait ArrayVisitor:
    """Trait for type-dispatched array operations.

    Implement this trait and call `visit(array, visitor)` to receive a
    concretely-typed array matching the runtime dtype of the Array.
    """

    fn visit_primitive[T: DataType](mut self, array: PrimitiveArray[T]): ...
    fn visit_string(mut self, array: StringArray): ...
    fn visit_list(mut self, array: ListArray): ...
    fn visit_struct(mut self, array: StructArray): ...


fn visit[V: ArrayVisitor](array: Array, mut visitor: V) raises:
    """Dispatch array to visitor based on its runtime dtype."""

    @parameter
    for dtype in [
        bool_,
        int8, int16, int32, int64,
        uint8, uint16, uint32, uint64,
        float16, float32, float64,
    ]:
        if array.dtype == materialize[dtype]():
            visitor.visit_primitive[dtype](PrimitiveArray[dtype](data=array.copy()))
            return

    if array.dtype.is_string():
        visitor.visit_string(StringArray(data=array.copy()))
    elif array.dtype.is_list():
        visitor.visit_list(ListArray(data=array.copy()))
    elif array.dtype.is_struct():
        visitor.visit_struct(StructArray(data=array.copy()))
    else:
        raise Error("visit: unsupported dtype {}".format(array.dtype))
