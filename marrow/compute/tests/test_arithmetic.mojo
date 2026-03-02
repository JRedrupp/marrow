from testing import assert_equal, assert_true, assert_false, TestSuite

from marrow.arrays import array, arange, Array, PrimitiveArray
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import int32, int64, float64, materialize
from marrow.compute.kernels.add import add


def test_add_typed():
    var a = array[int32]([1, 2, 3, 4])
    var b = array[int32]([10, 20, 30, 40])
    var result = add[int32](a, b)
    assert_equal(len(result), 4)
    assert_equal(result.unsafe_get(0), 11)
    assert_equal(result.unsafe_get(1), 22)
    assert_equal(result.unsafe_get(2), 33)
    assert_equal(result.unsafe_get(3), 44)


def test_add_with_nulls():
    """Nulls propagate: null + valid = null."""
    var a = PrimitiveBuilder[int32](3)
    a.unsafe_append(1)
    a.unsafe_append(2)
    # index 2 is null (bitmap zero-initialized, length set manually)
    a.length = 3

    var b = array[int32]([10, 20, 30])
    var result = add[int32](a^.freeze(), b)
    assert_equal(len(result), 3)
    assert_true(result.is_valid(0))
    assert_true(result.is_valid(1))
    assert_false(result.is_valid(2))
    assert_equal(result.unsafe_get(0), 11)
    assert_equal(result.unsafe_get(1), 22)


def test_add_length_mismatch():
    var a = array[int32]([1, 2])
    var b = array[int32]([1, 2, 3])
    try:
        _ = add[int32](a, b)
        assert_true(False, "should have raised")
    except:
        pass


def test_add_untyped():
    var a = Array(array[int64]([1, 2, 3]))
    var b = Array(array[int64]([4, 5, 6]))
    var result = add(a, b)
    assert_equal(result.length, 3)
    # Verify by downcasting back
    var typed = result.as_primitive[int64]()
    assert_equal(typed.unsafe_get(0), 5)
    assert_equal(typed.unsafe_get(1), 7)
    assert_equal(typed.unsafe_get(2), 9)


def test_add_empty():
    var a = array[int32]()
    var b = array[int32]()
    var result = add[int32](a, b)
    assert_equal(len(result), 0)


def test_add_float64():
    var a = array[float64]([1, 2, 3, 4])
    var b = array[float64]([10, 20, 30, 40])
    var result = add[float64](a, b)
    assert_equal(len(result), 4)
    assert_true(result.unsafe_get(0) == 11)
    assert_true(result.unsafe_get(1) == 22)
    assert_true(result.unsafe_get(2) == 33)
    assert_true(result.unsafe_get(3) == 44)


def test_add_large_array():
    """Exercise the SIMD fast path with an array larger than SIMD width."""
    var a = arange[int32](0, 1000)
    var b = arange[int32](0, 1000)
    var result = add[int32](a, b)
    assert_equal(len(result), 1000)
    assert_equal(result.unsafe_get(0), 0)
    assert_equal(result.unsafe_get(499), 998)
    assert_equal(result.unsafe_get(999), 1998)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
