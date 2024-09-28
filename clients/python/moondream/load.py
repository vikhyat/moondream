import struct
import ggml
import ctypes

from enum import IntEnum
from functools import reduce
from collections import OrderedDict

GGUF_MAGIC = 0x46554747
SUPPORTED_GGUF_VERSIONS = [3]
SUPPORTED_MODEL_ARCHITECTURES = ["md-vl-0"]


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12

    def format_str(self):
        format_map = {
            GGUFValueType.UINT8: "B",
            GGUFValueType.INT8: "b",
            GGUFValueType.UINT16: "H",
            GGUFValueType.INT16: "h",
            GGUFValueType.UINT32: "I",
            GGUFValueType.INT32: "i",
            GGUFValueType.FLOAT32: "f",
            GGUFValueType.BOOL: "?",
            GGUFValueType.UINT64: "Q",
            GGUFValueType.INT64: "q",
            GGUFValueType.FLOAT64: "d",
        }
        if self in format_map:
            return format_map[self]
        raise NotImplementedError(f"Non-scalar value type {self}")


class GGMLQuantizationType(IntEnum):
    F32 = 0
    F16 = 1


GGML_QUANT_SIZES: dict[GGMLQuantizationType, tuple[int, int]] = {
    GGMLQuantizationType.F32: (1, 4),
    GGMLQuantizationType.F16: (1, 2),
}


class CpuContextBuffer:
    def __init__(self, buffer_size: int = 256 * 1024 * 1024):
        self.buffer_size = buffer_size
        self._buffer = (ctypes.c_uint8 * self.buffer_size)()

    def resize(self, new_size: int):
        assert new_size > self.buffer_size

        self.buffer_size = new_size
        ctypes.resize(self._buffer, self.buffer_size)

    @property
    def buffer(self) -> ctypes.c_void_p:
        return ctypes.c_void_p(ctypes.addressof(self._buffer))


def read_str(fin):
    (key_len,) = struct.unpack("Q", (fin.read(struct.calcsize("Q"))))

    key = ""
    for _ in range(key_len):
        (key_c,) = struct.unpack("B", (fin.read(struct.calcsize("B"))))
        key += chr(key_c)

    return key


def read_value(fin, v_type):
    if v_type == GGUFValueType.STRING:
        return read_str(fin)
    elif v_type == GGUFValueType.ARRAY:
        (ary_type,) = struct.unpack("I", (fin.read(struct.calcsize("I"))))
        (ary_len,) = struct.unpack("Q", (fin.read(struct.calcsize("Q"))))
        value = []
        for _ in range(ary_len):
            value.append(read_value(fin, ary_type))
        return value
    else:
        fmt_str = GGUFValueType(v_type).format_str()
        (value,) = struct.unpack(fmt_str, (fin.read(struct.calcsize(fmt_str))))
        return value


def load_weights(gguf_file):
    with open(gguf_file, "rb") as fin:
        (magic,) = struct.unpack("i", (fin.read(struct.calcsize("i"))))
        assert magic == GGUF_MAGIC

        # TODO: handle systems with different endianness
        (gguf_version,) = struct.unpack("i", (fin.read(struct.calcsize("i"))))
        assert gguf_version in SUPPORTED_GGUF_VERSIONS

        tensor_count, kv_count = struct.unpack("qq", (fin.read(struct.calcsize("qq"))))

        # Load GGUF metadata key-values
        gguf_meta = {}
        for _ in range(kv_count):
            key = read_str(fin)
            (v_type,) = struct.unpack("I", (fin.read(struct.calcsize("I"))))
            value = read_value(fin, v_type)
            gguf_meta[key] = value

        # TODO: Add better error message here.
        assert gguf_meta["general.architecture"] in SUPPORTED_MODEL_ARCHITECTURES

        # Load tensor info
        tensor_info = OrderedDict()
        for _ in range(tensor_count):
            tensor_name = read_str(fin)
            tensor_dim_count: int = read_value(fin, GGUFValueType.UINT32)  # type: ignore [no-untyped-call]
            tensor_dims = struct.unpack(
                "Q" * tensor_dim_count,
                (fin.read(struct.calcsize("Q" * tensor_dim_count))),
            )
            tensor_dtype = read_value(fin, GGUFValueType.UINT32)
            tensor_offset = read_value(fin, GGUFValueType.UINT64)

            tensor_info[tensor_name] = (
                tensor_dim_count,
                tensor_dims,
                tensor_dtype,
                tensor_offset,
            )

        # Calculate weight buffer size
        # TODO: Figure out what we're missing in the calculation that makes us have
        # to add this constant value to the context size.
        ctx_size = 237000
        for _, dims, dtype, _ in tensor_info.values():
            n_elems = reduce(lambda x, y: x * y, dims)
            ctx_size += ggml.ggml_type_sizef(dtype) * n_elems
        ctx_size = int(ctx_size)

        # Load tensor weights.
        weights_buffer = CpuContextBuffer(ctx_size)
        init_params = ggml.ggml_init_params(
            mem_size=ctx_size,
            mem_buffer=weights_buffer.buffer,
            no_alloc=False,
        )
        ctx = ggml.ggml_init(init_params)
        if ctx is None:
            raise RuntimeError("Failed to initialize GGML context.")

        tensors = {}
        for tensor_name, (dim_count, dims, dtype, offset) in tensor_info.items():
            if dim_count == 1:
                tensor = ggml.ggml_new_tensor_1d(ctx, dtype, dims[0])
            elif dim_count == 2:
                tensor = ggml.ggml_new_tensor_2d(ctx, dtype, dims[0], dims[1])
            elif dim_count == 3:
                tensor = ggml.ggml_new_tensor_3d(ctx, dtype, dims[0], dims[1], dims[2])
            elif dim_count == 4:
                tensor = ggml.ggml_new_tensor_4d(
                    ctx, dtype, dims[0], dims[1], dims[2], dims[3]
                )
            else:
                raise NotImplementedError(
                    f"Unsupported tensor dimension count {dim_count} for {tensor_name}"
                )

            tensor_data = ggml.ggml_get_data(tensor)
            assert tensor_data is not None

            fin.readinto(
                (ctypes.c_uint8 * ggml.ggml_nbytes(tensor)).from_address(tensor_data)
            )

            tensors[tensor_name] = tensor

        return tensors
