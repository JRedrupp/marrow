from .arrays import *
from .schema import Schema


@fieldwise_init
struct RecordBatch:
    var schema: Schema
    var fields: List[Array]


# TODO: add Table which is a collection of chunked arrays with the same length, and a schema