# Changelog

## [Unreleased]

### Fixes

- **Nightly compat: compile errors against Mojo `1.0.0b3.dev2026080206`**
  (`marrow/{builders,kernels/hashtable,expr/relations}.mojo`, continuing the
  migration started in `28ad12f`..`7b4e978`): explicit `__deinit__` on
  `StructBuilder` to break the same mutually-recursive `Deinitable`
  auto-derivation cycle already fixed for `AnyBuilder`/`ArrayData` in
  `ab07e88`; explicit import of `Int32Type` in `kernels/hashtable.mojo`
  (implicit package-level re-export is now deprecated); two inline absolute
  `from marrow.x import Y` imports in `expr/relations.mojo` that created a
  second nominal identity for `AnyDataType`/`Schema` distinct from the
  relative imports used elsewhere in the file (same class of bug as
  `f149a0c`/`d2c92b5`), switched to relative imports; and five remaining
  `as_*` builder downcasts (`as_primitive`, `as_fixed_size_binary`,
  `as_year_month_interval`, `as_day_time_interval`,
  `as_month_day_nano_interval`) whose `ref[self._ptr[]]` return-type
  annotation was wider than the returned expression, narrowed to
  `ref[self._ptr[][T]]` matching the pattern established in `7b4e978`.

  Compiling `marrow/` (excluding `tests/` subdirectories — see note below)
  with `mojo package --Werror` at that point produced exactly one remaining
  error category: 11 errors in `marrow/views.mojo` from `elementwise`'s
  `Coord`-based signature and mandatory `DeviceContext` parameter, which no
  longer matched marrow's `IndexList`-based SIMD kernel callback type
  (`_apply_dispatch`, `_reduce_generator_wrapper` and their CPU-only call
  sites). That elementwise/`Coord`/mandatory-`DeviceContext` GPU API rework
  — first flagged in `34dc32a` — is now fixed; see the entry below.

- **`marrow/views.mojo`: migrate `elementwise`/`_reduce_generator_wrapper`
  call sites to the `Coord`-based, mandatory-`DeviceContext` MAX API**
  (resolves the 11 errors called out above, and completes the
  `34dc32a`-flagged rework): `elementwise`'s three overloads (`max.algorithm.
  functional`) dropped their `Optional[DeviceContext]` forms and their
  `IndexList`-taking closure convention in favor of a mandatory `context:
  DeviceContext` argument and a `Coord`-taking `process` closure; every CPU-
  only call site in this file (which previously omitted `context` entirely,
  relying on the now-removed optional overloads) now passes one explicitly.
  `DeviceContext`'s import also moved from `std.gpu.host` to `max.gpu.host`
  (which re-exports `get_gpu_target` too, so both come from one import now).

  Constructing that CPU-side `DeviceContext` is the interesting part:
  `DeviceContext()`'s `api` parameter defaults to `String(Self.
  default_device_info.api)`, which evaluates `GPUInfo.from_name[
  _accelerator_arch()]()` — on a machine with no accelerator this is a
  **compile-time** `comptime assert` failure (`constraint failed: Unknown
  GPU architecture detected`), not a runtime one, confirmed with a
  standalone probe. Passing `api="cpu"` explicitly skips evaluating that
  default and constructs successfully at runtime on CPU-only machines; a new
  `_cpu_device_context()` helper does this for every CPU-only `elementwise`
  call site. This isn't a guess — it mirrors what MAX's own CPU backend does
  internally (`max/mojo/max/algorithm/backend/cpu/parallelize.mojo`:
  `sync_parallelize` builds its context via `ctx.or_else(DeviceContext(
  api="cpu"))`).

  `_reduce_generator_wrapper`'s `output_fn`/`reduce_function` closures
  changed their SIMD-width parameter kind from `Int` to the new `SIMDLength`
  type (its `input_fn` closure kind and `IndexList`-based indexing were
  *not* changed, so only the output/combine sides needed updating); its
  `shape` argument changed from `IndexList[1](length)` to `Coord(length)`.
  `combine_capturing`'s `SIMDLength`→`Int` bridge to the (unchanged, still
  `Int`-parameterized) public `reduce[T, combine]`/`_reduce_dispatch`
  `combine` signature is a plain `Int(W)` conversion — verified compiling
  and running correctly via a standalone probe (`reduce[DType.int32, add](
  0..9) == 45`).

  One correctness-sensitive call site needed more than a mechanical
  signature update: `apply[In, op: BinaryFn[In, DType.bool]]` (comparison
  ops that bit-pack their `SIMD[bool, W]` result into a `BitmapView`) relied
  on the now-removed `use_blocking_impl=True` to force single-worker
  execution — its docstring already documented why: "bit-packed outputs
  need whole-byte-aligned stride to avoid scalar read-modify-write races
  between workers." `elementwise`'s own CPU parallel dispatch
  (`_elementwise_impl_cpu_1d` in `max/mojo/max/algorithm/backend/cpu/
  elementwise.mojo`) stripes work by `ceildiv(problem_size, num_workers)`,
  which is not guaranteed to be a multiple of 8 — so simply routing this
  call site through `elementwise`'s new auto-parallel CPU path (as every
  other, race-free call site in this file now does) would silently
  reintroduce that exact race above `elementwise`'s internal ~32768-element
  parallel threshold. Fixed by bypassing `elementwise` for this one CPU
  path entirely and running it single-threaded via `std.algorithm.
  vectorize` instead, preserving the original single-worker guarantee. The
  two byte-granular bitmap ops in this file (`process_zero`/
  `process_shifted`, indexed by whole output byte rather than by element)
  don't have this hazard — disjoint workers can never share a byte there —
  so those were left on `elementwise`'s auto-parallel CPU path.

  Net effect on `_apply_dispatch` (used by every element-strided
  `BufferView` `apply` overload): `ExecutionContext.serial()` no longer
  guarantees single-threaded execution once `length` crosses `elementwise`'s
  internal parallel threshold — `ctx.num_threads`/`ctx.wants_parallel` are
  no longer consulted on this path at all, since `use_blocking_impl` (the
  mechanism that respected them) was removed upstream. Not a correctness
  issue here (element-strided writes are disjoint regardless of chunk
  boundaries), but worth flagging: a caller invoking `apply` from inside its
  own already-parallel region can now nest with MAX's own CPU parallelism.
  `wants_parallel`/`resolved_num_threads`/`num_threads` remain in active use
  elsewhere in the tree (`kernels/sort.mojo`, `kernels/filter.mojo`,
  `kernels/join.mojo`, all via direct `sync_parallelize` striping, not
  `elementwise`) — this change does not make them dead code tree-wide, only
  unused on this one `views.mojo` path.

  Verified via `mojo package` against `marrow/` with `tests/` excluded
  (`--Werror`, target nightly `1.0.0b3.dev2026080206`): 0 errors, down from
  the 11 above. Not verified via marrow's own test suite — `pixi run -e dev
  pytest` remains blocked by the same `pontoneer` build-dependency failure
  noted elsewhere in this changelog, so nothing in this change has run
  against marrow's actual `test_views.mojo`/`test_views_gpu.mojo` coverage;
  standalone probes (outside the package, compiled and *run*, not just
  type-checked) were used to validate the `elementwise`/`_reduce_generator_
  wrapper`/`vectorize`/`DeviceContext` API usage patterns instead.

  **Known separate issue, not fixed in this pass:** `mojo package marrow`
  (marrow's own `pixi run package` task, and bison's `build-marrow` task)
  walks every `.mojo` file under the given directory unconditionally,
  including `marrow/tests/`, `marrow/kernels/tests/`, and
  `marrow/expr/tests/`. Those directories contain `def main()` entry points
  (by design — see `CLAUDE.md`'s `TestSuite.run` convention; they're built
  individually via `mojo build` by the pytest harness, never packaged), and
  the current nightly's `mojo package`/`mojo precompile` unconditionally
  rejects any `def main()` reachable under the packaged directory with
  `'main()' is not supported within packages`, regardless of `__init__.mojo`
  placement (confirmed with a minimal repro outside this package). Whether
  this is new nightly behavior or a long-unexercised path in marrow's own
  `package`/CI tasks was not determined. It affects `mojo package marrow`
  directly and therefore blocks bison's `build-marrow` pixi task, which runs
  the identical command against this fork.

  This defect also makes the *count* of errors from the literal
  `mojo package marrow --Werror` command an unreliable signal: on the
  unmodified `tests/`-inclusive tree, a single early parse failure in a
  `main()`-bearing test file appears to poison symbol resolution for
  sibling modules, producing large, non-representative secondary error
  counts (`unable to locate module 'marrow'`, `use of unknown declaration
  '...'`) that vary sharply — and non-monotonically with respect to real
  fixes — depending on unrelated changes elsewhere in the tree. Concretely,
  switching the two `expr/relations.mojo` imports above from absolute to
  relative (correct per the `f149a0c`/`d2c92b5` convention, and verified to
  fix the underlying identity-mismatch bug when compiling `marrow/` with
  `tests/` excluded) raises the *tests-inclusive* `--Werror` error count
  from 124 to ~872, isolated by bisection to that one change. The
  tests-excluded build is the reliable signal; the tests-inclusive count is
  not meaningful until the `main()`-in-package-root issue below is fixed.

  Fixing it requires relocating the three `tests/` directories out
  from under the `marrow/` package root (plus updating `pixi.toml`
  pytest paths, `conftest.py`, and any relative imports inside the moved
  test files) — a structural change with wide blast radius that could not be
  safely verified in this pass, since marrow's own `pixi run -e dev pytest`
  is currently blocked by the same `pontoneer` build-dependency failure
  noted for `pixi run -- mojo package` (see `progress.md`/task-4 report).
  Left as a follow-up task.

### Features

- **Sort kernel — `argsort` and `sort`** (`marrow/kernels/sort.mojo`):
  single-column sort for all array types. Primitive arrays use LSD radix sort
  (O(N), 8-bit passes, UInt64-encoded keys, float NaN/sign-bit transform) for
  N ≥ 32 768, with parallel histogram + scatter for N ≥ 524 288. PDQsort for
  N < 32 768 (faster on Apple M-series up to ~28K elements); insertion-sort
  leaf for N < 32. `BoolArray` uses O(N) counting sort; `StringArray` uses the
  Mojo stdlib comparison sort. Null partitioning (pre-sort bitmap scan) with
  `nulls_first`/`nulls_last` placement. `sort(StructArray, key_indices,
  ascending)` wraps `argsort` + `take` for multi-column sort.

- **Large binary, string, and list types** (`marrow/{dtypes,arrays,builders,ipc,c_data}.mojo`):
  added `LargeBinaryType`, `LargeStringType`, `LargeListType` (64-bit offsets);
  `BinaryLikeType` trait with `comptime offset: DType` and `StringLikeType` sub-trait
  for UTF-8 kernels; unified `BinaryArray[T: BinaryLikeType]` and
  `BinaryBuilder[T: BinaryLikeType]` with aliases `StringArray`, `LargeBinaryArray`,
  `LargeStringArray`, `StringBuilder`, `LargeBinaryBuilder`, `LargeStringBuilder`;
  IPC type codes 19/20/21 for large binary/utf8/list; C Data format codes `Z`/`U`/`+L`.

- **IPC support for dictionary-encoded columns** (`marrow/ipc.mojo`): the IPC
  file and stream writer now emits a `DictionaryBatch` message (header type 2)
  for each dictionary column before its first `RecordBatch`, encoding the
  column's value array as a separate body. The `RecordBatch` body carries only
  the integer indices. Dictionary blocks are registered in the IPC file footer so
  C++ / Rust / Go readers can locate them. The IPC reader detects
  `DictionaryEncoding` at schema-field slot 4, reconstructs `DictionaryType`
  (index type + value type + ordered flag), loads `DictionaryBatch` messages via
  footer-registered block offsets, and wires the decoded values back into
  `DictionaryArray` instances when reading record batches. Validated across all
  Arrow implementations (`dictionary` and `dictionary_unsigned` pass 14/14
  integration phases with C++, Rust, and Go).

- **Arrow interval types** (`marrow/{dtypes,scalars,arrays,builders,ipc,c_data}.mojo`, `python/`):
  added `IntervalType` trait and three concrete types — `YearMonthIntervalType` (int32, months),
  `DayTimeIntervalType` (int64, days+millis), `MonthDayNanoIntervalType` (int128, months+days+nanos).
  `AnyDataType` gains `is_interval()`, `is_year_month_interval()`, `is_day_time_interval()`,
  `is_month_day_nano_interval()` predicates and matching `as_*` accessors. Array, builder, and
  scalar aliases (`YearMonthIntervalArray/Builder/Scalar`, etc.) are fully wired into the
  `AnyArray`, `AnyBuilder`, and `AnyScalar` type-erased containers. C Data Interface uses
  format codes `tiM`, `tiD`, `tin`; IPC uses the `Interval` flatbuffer type with unit field.
  Python bindings expose `year_month_interval()`, `day_time_interval()`,
  `month_day_nano_interval()` factory functions.

- **Dictionary-encoded Arrow type** (`marrow/{dtypes,scalars,arrays,builders,
  c_data}.mojo`): added `DictionaryType` (index type + value type + ordered
  flag), `DictionaryScalar`, `DictionaryArray`, and `DictionaryBuilder`.
  `DictionaryArray.from_arrays(indices, values)` constructs from an integer
  indices array and an arbitrary values array; `__getitem__` decodes to the
  underlying value scalar; `slice()` is zero-copy. The C Data Interface emits
  the index type's format string and stores the value schema in the `dictionary`
  field of `CArrowSchema`, with `ARROW_FLAG_DICT_ORDERED = 1` when ordered;
  import detects a non-null `dictionary` field and reconstructs the type.
  Enables zero-copy exchange of PyArrow `DictionaryArray` via the Arrow C Data
  Interface (`__arrow_c_array__` / `__arrow_c_schema__` protocol).

- **Arrow Null type** (`marrow/{arrays,scalars,builders,ipc,c_data}.mojo`,
  `python/arrays.mojo`): added `NullArray`, `NullScalar`, `NullBuilder`
  (registered in the `AnyArray`, `AnyScalar`, `AnyBuilder` variants); IPC
  writer emits `Type.Null = 1` with zero body buffers; IPC reader skips the
  validity slot for null fields; C Data Interface uses `n_buffers = 0` for null
  per the spec; Python factory `ma.array(seq, type=ma.null())` builds a
  `NullArray` of the given length.

- **Fixed-size binary type** (`marrow/{dtypes,arrays,builders,ipc,c_data}.mojo`):
  added `FixedSizeBinaryType`, `FixedSizeBinaryArray`, `FixedSizeBinaryBuilder`;
  C Data format code `"w:<n>"`; IPC type code 15 (FixedSizeBinary).

- **Temporal array types** (`marrow/{dtypes,arrays,builders,ipc,c_data}.mojo`):
  `Date32Array`, `Date64Array`, `Time32Array`, `Time64Array`, `TimestampArray`,
  `DurationArray` with matching builders and type singletons; C Data format
  codes (`"tdD"`, `"tdm"`, `"tts"`, `"ttu"`, `"tsn:"`, `"tDn"`, etc.); IPC
  type codes and unit serialisation. Python constructors `ma.date32()`,
  `ma.date64()`, `ma.time32(unit)`, `ma.time64(unit)`, `ma.timestamp(unit)`,
  `ma.duration(unit)`.

- **Decimal types in C Data Interface and IPC**
  (`marrow/c_data.mojo`, `marrow/ipc.mojo`): wired `Decimal32Type`,
  `Decimal64Type`, `Decimal128Type`, `Decimal256Type` into schema export/import
  and IPC flatbuffer serialisation (precision, scale, bit-width).

- **Custom metadata round-trip via the C Data Interface**
  (`marrow/c_data.mojo`): `CArrowSchema.from_field` / `from_schema` now
  encode `Field.metadata` and `Schema.metadata` into the spec-defined
  metadata blob; `to_field` / `to_schema` decode it back. New
  `_encode_c_metadata` / `_decode_c_metadata` helpers handle the
  `int32 num_pairs ; (int32 key_len, key_bytes, int32 val_len, val_bytes)*`
  layout. `from_schema` now takes a full `Schema` rather than `List[Field]`
  so schema-level metadata flows through.

- **Per-field metadata** (`marrow/dtypes.mojo`, `python/dtypes.mojo`):
  `Field` carries an optional `metadata: Dict[String, String]`; the Python
  factory `ma.field(name, type, metadata={…})` accepts a dict; the C Data
  Interface and IPC flatbuffer encoder/decoder round-trip field-level
  key-value metadata.

- **Preserve nested-field names in IPC reader and C Data Interface**
  (`marrow/ipc.mojo`, `marrow/c_data.mojo`): the IPC `_read_field`
  decoder and the `CArrowSchema` list / fixed_size_list importer now preserve
  child Field names as-is, so Arrow files written by other implementations
  round-trip with the original schema.

- **Arrow IPC reader/writer** (`marrow/ipc.mojo`): `read_ipc_file()`,
  `write_ipc_file()`, `read_ipc_stream()`, `write_ipc_stream()`,
  `read_ipc_file_schema()`, `read_ipc_stream_schema()`, and streaming struct
  variants `RecordBatchFileReader`, `RecordBatchStreamReader`,
  `RecordBatchFileWriter`, `RecordBatchStreamWriter`. Supports all implemented
  Arrow types (bool, int8–64, uint8–64, float16/32/64, binary, utf8, list,
  fixed_size_list, struct, dictionary, null, temporal, decimal) with full
  nested and nullable column support. FlatBuffer encoding/decoding is a
  self-contained Rust-faithful port with correct soffset sign convention and
  `MetadataVersion::V5`.

- **GPU aggregate reductions** (`marrow/kernels/aggregate.mojo`):
  `sum_`, `min_`, `max_`, `product`, `any_`, `all_` now accept an
  `ExecutionContext`; when `.is_gpu()` is true the reduction runs as a
  single-pass GPU kernel via `_reduce_generator_wrapper`.

- **`ExecutionContext`** (`marrow/kernels/execution.mojo`): new struct bundling
  `num_threads` for CPU stripe parallelism and `device: Optional[DeviceContext]`
  for GPU. Implicit conversions from `Optional[DeviceContext]` and
  `DeviceContext` keep existing callers working. Factories: `.serial()`,
  `.parallel(num_threads=0)` (0 = `num_physical_cores()`), `.gpu(device)`.
  Wired through all kernels: arithmetic, aggregate, compare, filter, join, sort.

- **Partition-parallel hash join** (`marrow/kernels/join.mojo`,
  `marrow/kernels/hashtable.mojo`): `HashJoin` and `hash_join()` gain a
  `num_threads` argument. The parallel path radix-partitions both sides by the
  top bits of their hash into independent `SwissHashTable` instances, builds and
  probes them concurrently via `sync_parallelize`, and concatenates per-partition
  index pairs. No atomics on the hot path. At 10M×10M INNER join: **330 ms
  (serial) → 67 ms (parallel, 4.9× speedup)** — faster than Polars (97 ms),
  PyArrow (111 ms), and DuckDB (122 ms).

- **`RadixPartitioner`** (`marrow/kernels/hashtable.mojo`): partitions hashes +
  row indices by the top `num_bits` (default 6 → 64 partitions). Per-thread
  histogram → partition-major prefix sum → parallel scatter into shared flat
  buffers, then per-partition zero-copy slice via `ArcPointer`-shared immutable
  buffers.

- **Parallel per-column `take()`** (`marrow/kernels/filter.mojo`):
  `take[T](PrimitiveArray, indices, ctx)` and the `AnyArray` dispatcher
  accept an `ExecutionContext` and stripe the no-null fast path across workers.
  End-to-end 10M inner join assembly: **143 ms → 67 ms**.

- **Variant-based dispatch for `DataType`, `AnyArray`, and `Builder`**
  (`marrow/dtypes.mojo`, `marrow/arrays.mojo`, `marrow/builders.mojo`):
  Replaced integer-code dispatch with `Variant`-backed types using `comptime
  for` loops. Eliminates runtime `if`/`elif` chains across kernels, Python
  bindings, and the expression system.

- **`BoolArray` dedicated type** (`marrow/arrays.mojo`): bit-packed boolean
  arrays backed by a `Bitmap`, with `.values() -> BitmapView`, GPU transfer,
  and a matching `BoolBuilder`.

- **`BufferView` / `BitmapView` abstractions** (`marrow/views.mojo`):
  type-safe, non-owning views with `apply` dispatch, `compressed_store`,
  `pext`, and GPU-aware access.

- **`SwissHashTable`** (`marrow/kernels/hashtable.mojo`): open-addressing hash
  table with 7-bit control stamps, CSR chain storage, vectorised SIMD group
  matching, and a batch-build API.

- **Hash join** (`marrow/kernels/join.mojo`): `hash_join` kernel using
  `SwissHashTable` with separate build and probe phases.

- **`TestSuite` and `BenchSuite` framework** (`marrow/testing`):
  auto-discovery of `test_*` / `bench_*` functions via
  `__functions_in_module()`, with pytest harness integration, competition
  tables, and per-element throughput metrics.

- **AddressSanitizer support**: `pytest --asan` compiles test runners with ASAN
  instrumentation via `libcompiler-rt`.

- **GPU `BitmapView` and GPU rapidhash** (`marrow/kernels/`): `BitmapView`
  supports device-resident bitmaps; `rapidhash` ported to Metal/CUDA with
  128-bit multiply emulation.

- **Bounds checking** (`marrow/buffers.mojo`): `Buffer`, `Bitmap`, and
  `BufferView` accessors assert bounds in debug builds.

- **Unary math kernels** (`marrow/kernels/arithmetic.mojo`): `sign`, `sqrt`,
  `exp`, `exp2`, `log`, `log2`, `log10`, `log1p`, `floor`, `ceil`, `trunc`,
  `round`, `sin`, `cos` (floating-point), plus binary `pow_`, `floordiv`, `mod`.

- **Scalar types** (`marrow/scalars.mojo`): `PrimitiveScalar[T]`,
  `StringScalar`, `ListScalar`, `StructScalar`, `AnyScalar` — typed and
  type-erased scalar values mirroring the array hierarchy.

- **Group-by kernel** (`marrow/kernels/groupby.mojo`): fused
  `groupby(keys, values, aggregations)` that hashes, groups, and aggregates in
  a single pass. Supports `"sum"`, `"min"`, `"max"`, `"count"`, `"mean"`.
  Single-key (any primitive/string `AnyArray`) and multi-key (`StructArray`)
  grouping.

- **Hashing kernel** (`marrow/kernels/hashing.mojo`): `hash_` for primitive,
  string, and struct arrays; `hash_identity` for bool/uint8/int8.

- **Expression execution system** (`marrow/expr/`): pull-based streaming query
  executor with `col()`, `lit()`, `if_else()`, relational plan nodes
  (`InMemoryTable`, `Filter`, `Project`, `ParquetScan`, `Aggregate`), and
  `execute()` to collect `RecordBatch` results.

- **Parquet I/O** (`marrow/parquet.mojo`): `read_table(path)` and
  `write_table(table, path)` via the Arrow C Stream Interface.

- **Comparison kernels** (`marrow/kernels/compare.mojo`): `equal`,
  `not_equal`, `less`, `less_equal`, `greater`, `greater_equal` for typed and
  runtime-typed arrays; null-propagating; GPU variants available.

- **String kernels** (`marrow/kernels/string.mojo`): `string_lengths` returns
  byte lengths for each element.

- **RecordBatch column operations** (`marrow/tabular.mojo`): `slice`,
  `select`, `rename_columns`, `add_column`, `append_column`, `remove_column`,
  `set_column`, `to_struct_array`.

- **Table enhancements** (`marrow/tabular.mojo`): `Table.from_batches`,
  `Table.to_batches`, `Table.combine_chunks`.

- **Schema enhancements** (`marrow/schema.mojo`): `get_field_index`, `field`
  lookup by name, `names()`, equality operators, Python interop via Arrow C
  Data Interface.

- **Self-contained archery integration suite** (`integration/`, `pixi.toml`):
  `pixi run integration` clones apache/arrow + arrow-rs + arrow-go, builds all
  reference implementations, and runs cross-implementation tests against C++,
  Rust, Go, and Mojo. All four implementations pass: 119 cases across 14
  directional phases.
