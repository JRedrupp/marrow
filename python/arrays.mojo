from std.python import PythonObject, ConvertibleToPython, Python
from std.python.bindings import PythonModuleBuilder
from std.collections import OwnedKwargsDict
from std.python._cpython import CPython, PyObjectPtr, PyTypeObject, PyTypeObjectPtr
import marrow.arrays as arr
from marrow.builders import Builder
import marrow.builders as bld
import marrow.dtypes as dt


fn pymethod[
    T: AnyType,
    R: ConvertibleToPython,
    //,
    method: fn (T) -> R,
]() -> fn (UnsafePointer[T, MutAnyOrigin]) raises -> PythonObject:
    fn wrapper(ptr: UnsafePointer[T, MutAnyOrigin]) raises -> PythonObject:
        return method(ptr[]).to_python_object()

    return wrapper


# ---------------------------------------------------------------------------
# PyInferrer — mirrors PyArrow's PyInferrer (inference.cc)
# ---------------------------------------------------------------------------


struct PyInferrer(Copyable, Movable):
    """Infers the Arrow DataType of a Python sequence.

    Mirrors PyArrow's PyInferrer (inference.cc): counts occurrences by Python
    type in a single pass, recursing into list/dict elements via _visit_list and
    _visit_dict, then resolves to a DataType without re-iterating the sequence.
    """

    var none_count: Int
    var bool_count: Int
    var int_count: Int
    var float_count: Int
    var unicode_count: Int
    var unicode_bytes: Int
    var bytes_count: Int
    var list_count: Int
    var struct_count: Int

    # Child inferrers — List[PyInferrer] works because PyInferrer declares Copyable
    var _list_child: List[PyInferrer]      # 0 or 1 elements
    var _field_order: List[String]
    var _field_children: List[PyInferrer]  # parallel to _field_order

    # Cached CPython type pointers (obtained once at init)
    var _none_ptr: PyObjectPtr
    var _unicode_type: PyTypeObjectPtr  # PyUnicode_Type via lib.get_symbol
    var _bytes_type: PyTypeObjectPtr    # PyBytes_Type via lib.get_symbol
    var _list_type: PyTypeObjectPtr     # PyList_Type via lib.get_symbol
    var _tuple_type: PyTypeObjectPtr    # PyTuple_Type via lib.get_symbol
    var _dict_type: PyTypeObjectPtr     # cpython.PyDict_Type()

    fn __init__(out self) raises:
        self.none_count = 0
        self.bool_count = 0
        self.int_count = 0
        self.float_count = 0
        self.unicode_count = 0
        self.unicode_bytes = 0
        self.bytes_count = 0
        self.list_count = 0
        self.struct_count = 0
        self._list_child = []
        self._field_order = []
        self._field_children = []
        ref cpy = Python().cpython()
        self._none_ptr = cpy.Py_None()
        self._unicode_type = cpy.lib.get_symbol[PyTypeObject]("PyUnicode_Type")
        self._bytes_type = cpy.lib.get_symbol[PyTypeObject]("PyBytes_Type")
        self._list_type = cpy.lib.get_symbol[PyTypeObject]("PyList_Type")
        self._tuple_type = cpy.lib.get_symbol[PyTypeObject]("PyTuple_Type")
        self._dict_type = cpy.PyDict_Type()

    fn visit(mut self, element: PythonObject) raises:
        """Count one element's Python type, following PyArrow's Visit() order."""
        self.visit_ptr(element._obj_ptr)

    fn visit_ptr(mut self, ptr: PyObjectPtr) raises:
        """Count one element's Python type from a raw pointer."""
        ref cpy = Python().cpython()
        if cpy.Py_Is(ptr, self._none_ptr):
            self.none_count += 1
        elif cpy.PyBool_Check(ptr) != 0:  # exact bool check before PyLong_Check
            self.bool_count += 1
        elif cpy.PyFloat_Check(ptr) != 0:  # float before int
            self.float_count += 1
        elif cpy.PyLong_Check(ptr) != 0:
            self.int_count += 1
        elif cpy.PyObject_TypeCheck(ptr, self._unicode_type) != 0:
            self.unicode_count += 1
            self.unicode_bytes += len(cpy.PyUnicode_AsUTF8AndSize(ptr))
        elif cpy.PyObject_TypeCheck(ptr, self._bytes_type) != 0:
            self.bytes_count += 1
        elif cpy.PyObject_TypeCheck(ptr, self._dict_type) != 0:
            self.struct_count += 1
            self._visit_dict(PythonObject(from_borrowed=ptr))
        elif cpy.PyObject_TypeCheck(ptr, self._list_type) != 0:
            self.list_count += 1
            self._visit_list(PythonObject(from_borrowed=ptr))
        elif cpy.PyObject_TypeCheck(ptr, self._tuple_type) != 0:
            self.list_count += 1
            self._visit_list(PythonObject(from_borrowed=ptr))
        else:
            raise Error(
                "cannot include value of type: "
                + String(PythonObject(from_borrowed=ptr).__class__.__name__)
            )

    fn _visit_list(mut self, list_obj: PythonObject) raises:
        """Mirrors PyArrow's VisitSequence: recurse into list element's children."""
        if len(self._list_child) == 0:
            self._list_child.append(PyInferrer())
        for element in list_obj:
            self._list_child[0].visit(element)

    fn _visit_dict(mut self, dict_obj: PythonObject) raises:
        """Mirrors PyArrow's VisitDict: route each value to its field's child inferrer."""
        for key_obj in dict_obj.keys():
            var name = String(py=key_obj)
            var idx = -1
            for i in range(len(self._field_order)):
                if self._field_order[i] == name:
                    idx = i
                    break
            if idx == -1:
                idx = len(self._field_order)
                self._field_order.append(name)
                self._field_children.append(PyInferrer())
            self._field_children[idx].visit(dict_obj[key_obj])

    fn _total_count(self) -> Int:
        return (
            self.none_count
            + self.bool_count
            + self.int_count
            + self.float_count
            + self.unicode_count
            + self.bytes_count
            + self.list_count
            + self.struct_count
        )

    fn _get_binary_type(self) raises -> dt.DataType:
        if self.bytes_count + self.none_count != self._total_count():
            raise Error("cannot mix bytes and non-bytes values")
        return dt.binary

    fn _get_list_type(self) raises -> dt.DataType:
        if self.list_count + self.none_count != self._total_count():
            raise Error("cannot mix list and non-list values")
        if len(self._list_child) == 0:
            raise Error("cannot infer type: all-null list")
        return dt.list_(self._list_child[0]._get_type())

    fn _get_struct_type(self) raises -> dt.DataType:
        if self.struct_count + self.none_count != self._total_count():
            raise Error("cannot mix dict and non-dict values")
        var fields: List[dt.Field] = []
        for i in range(len(self._field_order)):
            var child_dtype = self._field_children[i]._get_type()
            fields.append(
                dt.Field(self._field_order[i], child_dtype, nullable=True)
            )
        return dt.struct_(fields)

    fn _get_primitive_type(self) raises -> dt.DataType:
        if self.unicode_count > 0 and (
            self.bool_count + self.int_count + self.float_count
        ) > 0:
            raise Error("cannot mix string and numeric types")
        if self.float_count > 0:
            return dt.float64
        if self.int_count > 0:
            return dt.int64
        if self.bool_count > 0:
            return dt.bool_
        if self.unicode_count > 0:
            return dt.string
        return dt.null  # empty sequence or all-None

    fn _get_type(self) raises -> dt.DataType:
        if self.bytes_count > 0:
            return self._get_binary_type()
        if self.list_count > 0:
            return self._get_list_type()
        if self.struct_count > 0:
            return self._get_struct_type()
        return self._get_primitive_type()

    fn infer(mut self, obj: PythonObject) raises -> dt.DataType:
        """Single pass: visit all elements, then resolve to a DataType."""
        ref cpy = Python().cpython()
        var list_ptr = obj._obj_ptr
        var n = len(obj)
        for i in range(n):
            var item_ptr = cpy.PyList_GetItem(list_ptr, i)
            self.visit_ptr(item_ptr)
        return self._get_type()


# ---------------------------------------------------------------------------
# PyList — lightweight helper for fast CPython list access
# ---------------------------------------------------------------------------


struct PyList:
    """Zero-overhead access to CPython list internals via borrowed references."""

    var _ptr: PyObjectPtr
    var _n: Int
    var _none: PyObjectPtr

    fn __init__(out self, obj: PythonObject) raises:
        ref cpy = Python().cpython()
        self._ptr = obj._obj_ptr
        self._n = len(obj)
        self._none = cpy.Py_None()

    @always_inline
    fn __len__(self) -> Int:
        return self._n

    @always_inline
    fn get(self, i: Int) -> PyObjectPtr:
        """Get item at index i (borrowed reference)."""
        return Python().cpython().PyList_GetItem(self._ptr, i)

    @always_inline
    fn is_none(self, item: PyObjectPtr) -> Bool:
        return Python().cpython().Py_Is(item, self._none)

    @always_inline
    fn as_int(self, item: PyObjectPtr) -> Int:
        return Python().cpython().PyLong_AsSsize_t(item)

    @always_inline
    fn as_float(self, item: PyObjectPtr) -> Float64:
        return Python().cpython().PyFloat_AsDouble(item)


# ---------------------------------------------------------------------------
# PyConverter — Python-to-Arrow converter mirroring PyArrow's Converter
# ---------------------------------------------------------------------------


struct PyConverter(Copyable, Movable):
    """Python-to-Arrow converter. Mirrors PyArrow's Converter hierarchy.

    Public API: extend(sequence), append(value), finish() → PythonObject.
    Type-specific helpers (_extend_primitive, _extend_string, etc.) are separated
    for clarity, mirroring how PyArrow uses separate Converter subclasses per type.
    """

    var builder: Builder
    var has_nulls: Bool
    var total_bytes: Int

    fn __init__(
        out self,
        dtype: dt.DataType,
        capacity: Int = 0,
        has_nulls: Bool = True,
        total_bytes: Int = 0,
    ) raises:
        self.builder = Builder(dtype, capacity)
        self.has_nulls = has_nulls
        self.total_bytes = total_bytes

    fn __init__(out self, var builder: Builder, has_nulls: Bool = True):
        self.builder = builder^
        self.has_nulls = has_nulls
        self.total_bytes = 0

    # ── public dispatch ─────────────────────────────────────────────────────

    fn extend(mut self, obj: PythonObject) raises:
        """Append a full Python sequence. Dispatches once to a type-specific path."""
        var dtype = self.builder.dtype()
        comptime for T in dt.primitive_dtypes:
            if dtype == T:
                self._extend_primitive[T](obj)
                return
        if dtype.is_string():
            self._extend_string(obj)
        elif dtype.is_binary():
            raise Error("binary array construction not yet supported")
        elif dtype.is_list():
            self._extend_list(obj)
        elif dtype.is_struct():
            self._extend_struct(obj)
        else:
            raise Error("unsupported type: " + String(dtype))

    fn append(mut self, value: PythonObject) raises:
        """Append a single Python value."""
        var dtype = self.builder.dtype()
        comptime for T in dt.primitive_dtypes:
            if dtype == T:
                self._append_primitive[T](value)
                return
        if dtype.is_string():
            self._append_string(value)
        elif dtype.is_list():
            self._append_list(value)
        elif dtype.is_struct():
            self._append_struct(value)
        else:
            raise Error("unsupported type: " + String(dtype))

    fn finish(mut self) raises -> PythonObject:
        return self.builder.finish().to_python_object()

    # ── type-specific extend paths ───────────────────────────────────────────

    fn _extend_primitive[T: dt.DataType](mut self, obj: PythonObject) raises:
        var pb = self.builder.as_primitive[T]()
        ref cpy = Python().cpython()
        var n = len(obj)
        var list_ptr = obj._obj_ptr
        var none_ptr = cpy.Py_None()
        pb.reserve(n)
        if self.has_nulls:
            for i in range(n):
                var item = cpy.PyList_GetItem(list_ptr, i)
                if cpy.Py_Is(item, none_ptr):
                    pb.unsafe_append_null()
                else:
                    comptime if T.native.is_floating_point():
                        pb.unsafe_append(Scalar[T.native](cpy.PyFloat_AsDouble(item)))
                    else:
                        pb.unsafe_append(Scalar[T.native](cpy.PyLong_AsSsize_t(item)))
        else:
            for i in range(n):
                var item = cpy.PyList_GetItem(list_ptr, i)
                comptime if T.native.is_floating_point():
                    pb.unsafe_append(Scalar[T.native](cpy.PyFloat_AsDouble(item)))
                else:
                    pb.unsafe_append(Scalar[T.native](cpy.PyLong_AsSsize_t(item)))

    fn _extend_string(mut self, obj: PythonObject) raises:
        var sb = self.builder.as_string()
        ref cpy = Python().cpython()
        var n = len(obj)
        var list_ptr = obj._obj_ptr
        var none_ptr = cpy.Py_None()

        sb.reserve(n)
        # Pre-compute total bytes if not already known from inference.
        var total_bytes = self.total_bytes
        if total_bytes == 0:
            for i in range(n):
                var item = cpy.PyList_GetItem(list_ptr, i)
                if not cpy.Py_Is(item, none_ptr):
                    total_bytes += len(cpy.PyUnicode_AsUTF8AndSize(item))
        sb.reserve_bytes(total_bytes)

        if self.has_nulls:
            for i in range(n):
                var item = cpy.PyList_GetItem(list_ptr, i)
                if cpy.Py_Is(item, none_ptr):
                    sb.unsafe_append_null()
                else:
                    var s = cpy.PyUnicode_AsUTF8AndSize(item)
                    sb.unsafe_append(s.unsafe_ptr(), len(s))
        else:
            for i in range(n):
                var item = cpy.PyList_GetItem(list_ptr, i)
                var s = cpy.PyUnicode_AsUTF8AndSize(item)
                sb.unsafe_append(s.unsafe_ptr(), len(s))

    fn _extend_list(mut self, obj: PythonObject) raises:
        var lb = self.builder.as_list()
        var child = PyConverter(self.builder.child(0))
        for element in obj:
            if element is None:
                lb.append_null()
            else:
                child.extend(element)
                lb.append(True)

    fn _extend_struct(mut self, obj: PythonObject) raises:
        var dtype = self.builder.dtype()
        var sb = self.builder.as_struct()
        var children = List[PyConverter]()
        for i in range(len(dtype.fields)):
            children.append(
                PyConverter(self.builder.child(i))
            )
        for element in obj:
            var is_null = element is None
            for i in range(len(dtype.fields)):
                if is_null:
                    children[i].append(PythonObject(None))
                else:
                    children[i].append(element.get(dtype.fields[i].name))
            sb.append(not is_null)

    # ── type-specific append paths ───────────────────────────────────────────

    fn _append_primitive[T: dt.DataType](mut self, value: PythonObject) raises:
        var pb = self.builder.as_primitive[T]()
        if value is None:
            pb.append_null()
        else:
            comptime if T == dt.bool_:
                pb.append(value.__bool__())
            else:
                pb.append(Scalar[T.native](py=value))

    fn _append_string(mut self, value: PythonObject) raises:
        var sb = self.builder.as_string()
        if value is None:
            sb.append_null()
        else:
            sb.append(String(py=value))

    fn _append_list(mut self, value: PythonObject) raises:
        var lb = self.builder.as_list()
        if value is None:
            lb.append_null()
        else:
            var child = PyConverter(self.builder.child(0))
            child.extend(value)
            lb.append(True)

    fn _append_struct(mut self, value: PythonObject) raises:
        var dtype = self.builder.dtype()
        var sb = self.builder.as_struct()
        var is_null = value is None
        for i in range(len(dtype.fields)):
            var child = PyConverter(self.builder.child(i))
            if is_null:
                child.append(PythonObject(None))
            else:
                child.append(value.get(dtype.fields[i].name))
        sb.append(not is_null)


# ---------------------------------------------------------------------------
# Public Python functions
# ---------------------------------------------------------------------------


fn infer_type(obj: PythonObject) raises -> PythonObject:
    var inferrer = PyInferrer()
    return inferrer.infer(obj).to_python_object()


fn array(
    obj: PythonObject, kwargs: OwnedKwargsDict[PythonObject]
) raises -> PythonObject:
    var dtype: dt.DataType
    var has_nulls = True
    var total_bytes = 0
    if opt := kwargs.find("type"):
        dtype = opt.value().downcast_value_ptr[dt.DataType]()[]
    else:
        var inferrer = PyInferrer()
        dtype = inferrer.infer(obj)
        has_nulls = inferrer.none_count > 0
        total_bytes = inferrer.unicode_bytes

    if dtype.is_null():
        raise Error(
            "cannot build array: sequence is empty or all-None"
            " (provide type= explicitly)"
        )

    var builder = PyConverter(dtype, len(obj), has_nulls, total_bytes)
    builder.extend(obj)
    return builder.finish()


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Add array types and constructors to the Python API."""

    _ = (
        mb.add_type[arr.BoolArray]("BoolArray")
        .def_method[pymethod[arr.BoolArray.__len__]()]("__len__")
        .def_method[pymethod[arr.BoolArray.null_count]()]("null_count")
    )
    _ = (
        mb.add_type[arr.Int8Array]("Int8Array")
        .def_method[pymethod[arr.Int8Array.__len__]()]("__len__")
        .def_method[pymethod[arr.Int8Array.null_count]()]("null_count")
    )
    _ = (
        mb.add_type[arr.Int16Array]("Int16Array")
        .def_method[pymethod[arr.Int16Array.__len__]()]("__len__")
        .def_method[pymethod[arr.Int16Array.null_count]()]("null_count")
    )
    _ = (
        mb.add_type[arr.Int32Array]("Int32Array")
        .def_method[pymethod[arr.Int32Array.__len__]()]("__len__")
        .def_method[pymethod[arr.Int32Array.null_count]()]("null_count")
    )
    _ = (
        mb.add_type[arr.Int64Array]("Int64Array")
        .def_method[pymethod[arr.Int64Array.__len__]()]("__len__")
        .def_method[pymethod[arr.Int64Array.null_count]()]("null_count")
    )
    _ = (
        mb.add_type[arr.UInt8Array]("UInt8Array")
        .def_method[pymethod[arr.UInt8Array.__len__]()]("__len__")
        .def_method[pymethod[arr.UInt8Array.null_count]()]("null_count")
    )
    _ = (
        mb.add_type[arr.UInt16Array]("UInt16Array")
        .def_method[pymethod[arr.UInt16Array.__len__]()]("__len__")
        .def_method[pymethod[arr.UInt16Array.null_count]()]("null_count")
    )
    _ = (
        mb.add_type[arr.UInt32Array]("UInt32Array")
        .def_method[pymethod[arr.UInt32Array.__len__]()]("__len__")
        .def_method[pymethod[arr.UInt32Array.null_count]()]("null_count")
    )
    _ = (
        mb.add_type[arr.UInt64Array]("UInt64Array")
        .def_method[pymethod[arr.UInt64Array.__len__]()]("__len__")
        .def_method[pymethod[arr.UInt64Array.null_count]()]("null_count")
    )
    _ = (
        mb.add_type[arr.Float32Array]("Float32Array")
        .def_method[pymethod[arr.Float32Array.__len__]()]("__len__")
        .def_method[pymethod[arr.Float32Array.null_count]()]("null_count")
    )
    _ = (
        mb.add_type[arr.Float64Array]("Float64Array")
        .def_method[pymethod[arr.Float64Array.__len__]()]("__len__")
        .def_method[pymethod[arr.Float64Array.null_count]()]("null_count")
    )
    _ = (
        mb.add_type[arr.StringArray]("StringArray")
        .def_method[pymethod[arr.StringArray.__len__]()]("__len__")
        .def_method[pymethod[arr.StringArray.null_count]()]("null_count")
    )
    _ = (
        mb.add_type[arr.ListArray]("ListArray")
        .def_method[pymethod[arr.ListArray.__len__]()]("__len__")
        .def_method[pymethod[arr.ListArray.null_count]()]("null_count")
    )
    _ = mb.add_type[arr.FixedSizeListArray]("FixedSizeListArray")
    _ = (
        mb.add_type[arr.StructArray]("StructArray")
        .def_method[pymethod[arr.StructArray.__len__]()]("__len__")
        .def_method[pymethod[arr.StructArray.null_count]()]("null_count")
    )

    mb.def_function[infer_type](
        "infer_type", docstring="Infer the Arrow type of a Python sequence."
    )
    mb.def_function[array](
        "array", docstring="Create a marrow array from a Python sequence."
    )
