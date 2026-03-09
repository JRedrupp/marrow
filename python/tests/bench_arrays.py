"""Benchmarks for marrow.array() vs PyArrow."""

import pyarrow as pa
import marrow as ma


N = 10000


int_data = list(range(N))
float_data = [float(i) for i in range(N)]
string_data = [f"item-{i}" for i in range(N)]
int_data_with_nulls = [i if i % 10 != 0 else None for i in range(N)]
float_data_with_nulls = [float(i) if i % 10 != 0 else None for i in range(N)]
string_data_with_nulls = [f"item-{i}" if i % 10 != 0 else None for i in range(N)]
nested_list_data = [[i, i + 1, i + 2] for i in range(N // 3)]
struct_data = [{"x": i, "y": float(i), "z": f"s{i}"} for i in range(N // 3)]


# ── int64 ──────────────────────────────────────────────────────────────────


def test_marrow_int64(benchmark):
    benchmark(ma.array, int_data, type=ma.int64())


def test_pyarrow_int64(benchmark):
    benchmark(pa.array, int_data, type=pa.int64())


def test_marrow_int64_nulls(benchmark):
    benchmark(ma.array, int_data_with_nulls, type=ma.int64())


def test_pyarrow_int64_nulls(benchmark):
    benchmark(pa.array, int_data_with_nulls, type=pa.int64())


# ── float64 ────────────────────────────────────────────────────────────────


def test_marrow_float64(benchmark):
    benchmark(ma.array, float_data, type=ma.float64())


def test_pyarrow_float64(benchmark):
    benchmark(pa.array, float_data, type=pa.float64())


def test_marrow_float64_nulls(benchmark):
    benchmark(ma.array, float_data_with_nulls, type=ma.float64())


def test_pyarrow_float64_nulls(benchmark):
    benchmark(pa.array, float_data_with_nulls, type=pa.float64())


# ── string ─────────────────────────────────────────────────────────────────


def test_marrow_string(benchmark):
    benchmark(ma.array, string_data, type=ma.string())


def test_pyarrow_string(benchmark):
    benchmark(pa.array, string_data, type=pa.string())


def test_marrow_string_nulls(benchmark):
    benchmark(ma.array, string_data_with_nulls, type=ma.string())


def test_pyarrow_string_nulls(benchmark):
    benchmark(pa.array, string_data_with_nulls, type=pa.string())


# ── infer (no explicit type) ──────────────────────────────────────────────


def test_marrow_int64_infer(benchmark):
    benchmark(ma.array, int_data)


def test_pyarrow_int64_infer(benchmark):
    benchmark(pa.array, int_data)


def test_marrow_string_infer(benchmark):
    benchmark(ma.array, string_data)


def test_pyarrow_string_infer(benchmark):
    benchmark(pa.array, string_data)


# ── nested list ────────────────────────────────────────────────────────────


def test_marrow_nested_list(benchmark):
    benchmark(ma.array, nested_list_data)


def test_pyarrow_nested_list(benchmark):
    benchmark(pa.array, nested_list_data)


# ── struct ─────────────────────────────────────────────────────────────────


def test_marrow_struct(benchmark):
    benchmark(ma.array, struct_data)


def test_pyarrow_struct(benchmark):
    benchmark(pa.array, struct_data)
