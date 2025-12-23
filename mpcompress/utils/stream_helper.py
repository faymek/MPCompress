# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import struct
from pathlib import Path


def filesize(filepath: str) -> int:
    """
    Get the size of a file in bytes.

    Args:
        filepath (str): Path to the file.

    Returns:
        size (int): File size in bytes.

    Raises:
        ValueError: If the file does not exist or is not a regular file.
    """
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def write_uints(fd, values, fmt=">{:d}I"):
    """
    Write unsigned integers to a file descriptor.

    Args:
        fd (file): File descriptor to write to.
        values (list): Sequence of unsigned integers to write.
        fmt (str): Format string for struct.pack. Defaults to ">{:d}I" (big-endian).

    Returns:
        bytes_written (int): Number of bytes written (4 bytes per integer).
    """
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4


def write_uchars(fd, values, fmt=">{:d}B"):
    """
    Write unsigned characters (bytes) to a file descriptor.

    Args:
        fd (file): File descriptor to write to.
        values (list): Sequence of unsigned characters (0-255) to write.
        fmt (str): Format string for struct.pack. Defaults to ">{:d}B" (big-endian).

    Returns:
        bytes_written (int): Number of bytes written (1 byte per character).
    """
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values)


def read_uints(fd, n, fmt=">{:d}I"):
    """
    Read unsigned integers from a file descriptor.

    Args:
        fd (file): File descriptor to read from.
        n (int): Number of unsigned integers to read.
        fmt (str): Format string for struct.unpack. Defaults to ">{:d}I" (big-endian).

    Returns:
        out_uints (tuple): Tuple of n unsigned integers.
    """
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    """
    Read unsigned characters (bytes) from a file descriptor.

    Args:
        fd (file): File descriptor to read from.
        n (int): Number of unsigned characters to read.
        fmt (str): Format string for struct.unpack. Defaults to ">{:d}B" (big-endian).

    Returns:
        out_chars (tuple): Tuple of n unsigned characters (0-255).
    """
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    """
    Write bytes to a file descriptor.

    Args:
        fd (file): File descriptor to write to.
        values (bytes): Bytes to write.
        fmt (str): Format string for struct.pack. Defaults to ">{:d}s" (big-endian).

    Returns:
        bytes_written (int): Number of bytes written, or 0 if values is empty.
    """
    if len(values) == 0:
        return 0
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values)


def read_bytes(fd, n, fmt=">{:d}s"):
    """
    Read bytes from a file descriptor.

    Args:
        fd (file): File descriptor to read from.
        n (int): Number of bytes to read.
        fmt (str): Format string for struct.unpack. Defaults to ">{:d}s" (big-endian).

    Returns:
        out_bytes (bytes): The read bytes.
    """
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def write_ushorts(fd, values, fmt=">{:d}H"):
    """
    Write unsigned short integers (16-bit) to a file descriptor.

    Args:
        fd (file): File descriptor to write to.
        values (list): Sequence of unsigned short integers to write.
        fmt (str): Format string for struct.pack. Defaults to ">{:d}H" (big-endian).

    Returns:
        bytes_written (int): Number of bytes written (2 bytes per short).
    """
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 2


def read_ushorts(fd, n, fmt=">{:d}H"):
    """
    Read unsigned short integers (16-bit) from a file descriptor.

    Args:
        fd (file): File descriptor to read from.
        n (int): Number of unsigned shorts to read.
        fmt (str): Format string for struct.unpack. Defaults to ">{:d}H" (big-endian).

    Returns:
        out_ushorts (tuple): Tuple of n unsigned short integers.
    """
    sz = struct.calcsize("H")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_uint_adaptive(f, a):
    """
    Write an unsigned integer using adaptive-length encoding.

    The encoding uses variable-length representation:
    - 1 byte for values < 128 (7 bits)
    - 2 bytes for values < 16384 (14 bits)
    - 4 bytes for values < 1073741824 (30 bits)

    Args:
        f (file): File descriptor to write to.
        a (int): Unsigned integer to write (must be < 2^30).

    Returns:
        bytes_written (int): Number of bytes written (1, 2, or 4).
    """
    if a < (1 << 7):
        a0 = (a >> 0) & 0xFF
        a0 = a0 | (0x00 << 7)
        write_uchars(f, (a0,))
        return 1

    if a < (1 << 14):
        a0 = (a >> 0) & 0xFF
        a1 = (a >> 8) & 0xFF
        a1 = a1 | (0x02 << 6)
        write_uchars(f, (a1, a0))
        return 2

    assert a < (1 << 30)
    a0 = (a >> 0) & 0xFF
    a1 = (a >> 8) & 0xFF
    a2 = (a >> 16) & 0xFF
    a3 = (a >> 24) & 0xFF
    a3 = a3 | (0x03 << 6)
    write_uchars(f, (a3, a2, a1, a0))
    return 4


def read_uint_adaptive(f):
    """
    Read an unsigned integer using adaptive-length decoding.

    Decodes integers written by write_uint_adaptive. The encoding uses
    variable-length representation based on the value size.

    Args:
        f (file): File descriptor to read from.

    Returns:
        out_value (int): The decoded unsigned integer.
    """
    a3 = read_uchars(f, 1)[0]
    if (a3 >> 7) == 0:
        return a3

    a2 = read_uchars(f, 1)[0]

    if (a3 >> 6) == 0x02:
        a3 = a3 & 0x3F
        return (a3 << 8) + a2
    a3 = a3 & 0x3F
    a1 = read_uchars(f, 1)[0]
    a0 = read_uchars(f, 1)[0]
    return (a3 << 24) + (a2 << 16) + (a1 << 8) + a0


class NalType(enum.IntEnum):
    """
    Network Abstraction Layer (NAL) unit types.

    Attributes:
        NAL_SPS (int): Sequence Parameter Set type (0).
        NAL_I (int): Intra-coded frame type (1).
        NAL_P (int): Predictive-coded frame type (2).
    """

    NAL_SPS = 0
    NAL_I = 1
    NAL_P = 2


class SpsManager:
    """
    Manager for Sequence Parameter Sets (SPS).

    Maintains a list of SPS entries and provides methods to find, reuse,
    update, and insert SPS configurations.
    """

    def __init__(self):
        """
        Initialize an empty SPS manager.
        """
        super().__init__()
        self.sps_list = []

    def find_identical_sps(self, target_sps):
        """
        Find an SPS with identical parameters to the target.

        Args:
            target_sps (dict): SPS dictionary to match against.

        Returns:
            sps (dict or None): A copy of the matching SPS if found, None otherwise.
        """
        for sps in self.sps_list:
            if (
                sps["height"] == target_sps["height"]
                and sps["width"] == target_sps["width"]
                and sps["use_ada_i"] == target_sps["use_ada_i"]
                and sps["ec_part"] == target_sps["ec_part"]
            ):
                return sps.copy()
        return None

    def resue_or_insert(self, sps):
        """
        Reuse an identical SPS if found, otherwise insert a new one.

        If an identical SPS exists (based on height, width, use_ada_i, ec_part),
        returns the existing SPS. Otherwise, assigns a new sps_id and inserts it.

        Args:
            sps (dict): SPS dictionary to reuse or insert.

        Returns:
            sps (dict): The SPS dictionary (existing or new).
            inserted (bool): True if a new SPS was inserted, False if reused.
        """
        ref_sps = self.find_identical_sps(sps)
        if ref_sps is None:
            new_sps = sps.copy()
            new_sps["sps_id"] = len(self.sps_list)
            self.sps_list.append(new_sps)
            return new_sps, True
        else:
            return ref_sps, False

    def update_or_insert(self, sps):
        """
        Update an existing SPS or insert a new one.

        If an SPS with the same sps_id exists, it is updated. Otherwise,
        a new SPS is appended to the list.

        Args:
            sps (dict): SPS dictionary to update or insert.
        """
        for i in range(len(self.sps_list)):
            if self.sps_list[i]["sps_id"] == sps["sps_id"]:
                self.sps_list[i] = sps.copy()
                return
        self.sps_list.append(sps.copy())

    def find_sps_by_id(self, sps_id):
        """
        Find an SPS by its ID.

        Args:
            sps_id (int): The SPS ID to search for.

        Returns:
            sps (dict or None): The SPS dictionary if found, None otherwise.
        """
        for sps in self.sps_list:
            if sps["sps_id"] == sps_id:
                return sps
        return None


def write_sps(fd, sps):
    """
    Write a Sequence Parameter Set (SPS) to a file descriptor.

    Format:
        - nal_type(4 bits), sps_id(4 bits) - 1 byte
        - height (variable length)
        - width (variable length)
        - reserved(6 bits), ec_part(1 bit), use_ada_i(1 bit) - 1 byte

    Args:
        fd (file): File descriptor to write to.
        sps (dict): SPS dictionary containing:
            - sps_id (int): SPS ID (0-15)
            - height (int): Frame height
            - width (int): Frame width
            - ec_part (int): EC partition flag (0-1)
            - use_ada_i (int): Adaptive I-frame flag (0-1)

    Returns:
        bytes_written (int): Number of bytes written.

    Raises:
        AssertionError: If sps_id is not in [0, 16) or use_ada_i is not in [0, 2).
    """
    assert 0 <= sps["sps_id"] < 16
    assert 0 <= sps["use_ada_i"] < 2
    written = 0
    flag = int((NalType.NAL_SPS << 4) + sps["sps_id"])
    written += write_uchars(fd, [flag])
    written += write_uint_adaptive(fd, sps["height"])
    written += write_uint_adaptive(fd, sps["width"])
    flag = (sps["ec_part"] << 2) + sps["use_ada_i"]
    written += write_uchars(fd, [flag])
    return written


def read_vps(fd):
    """
    Read a Video Parameter Set (VPS) header from a file descriptor.

    For NAL types < 3, reads a simple header with nal_type and sps_id.
    For other NAL types, reads frame_num and associated sps_ids.

    Args:
        fd (file): File descriptor to read from.

    Returns:
        header (dict): Dictionary containing the nal_type, sps_id, frame_num, and sps_ids.
    """
    header = {}
    flag = read_uchars(fd, 1)[0]
    nal_type = flag >> 4
    header["nal_type"] = NalType(nal_type)
    if nal_type < 3:
        header["sps_id"] = flag & 0x0F
        return header

    frame_num_minus1 = flag & 0x0F
    frame_num = frame_num_minus1 + 1
    header["frame_num"] = frame_num
    sps_ids = []
    for _ in range(0, frame_num, 2):
        flag = read_uchars(fd, 1)[0]
        sps_ids.append(flag >> 4)
        sps_ids.append(flag & 0x0F)
    sps_ids = sps_ids[:frame_num]
    header["sps_ids"] = sps_ids
    return header


def read_sps_remaining(fd, sps_id):
    """
    Read the remaining SPS fields from a file descriptor.

    Reads height, width, and flags (ec_part, use_ada_i) after the initial
    header has been read. The sps_id is provided as a parameter.

    Args:
        fd (file): File descriptor to read from.
        sps_id (int): The SPS ID (already read from header).

    Returns:
        sps (dict): SPS dictionary containing the sps_id, height, width, ec_part, and use_ada_i.
    """
    sps = {}
    sps["sps_id"] = sps_id
    sps["height"] = read_uint_adaptive(fd)
    sps["width"] = read_uint_adaptive(fd)
    flag = read_uchars(fd, 1)[0]
    sps["ec_part"] = (flag >> 2) & 0x01
    sps["use_ada_i"] = flag & 0x01
    return sps


def write_picture(fd, is_i_frame, sps_id, qp, bit_stream):
    """
    Write a picture (frame) to a file descriptor.

    Format:
        - nal_type(4 bits), sps_id(4 bits) - 1 byte
        - qp (quantization parameter) - 1 byte
        - bit_stream length (variable length)
        - bit_stream data

    Note: Since all streams are written to the same file, the per-frame
    length must be written. If frames were packed independently, this
    would not be necessary.

    Args:
        fd (file): File descriptor to write to.
        is_i_frame (bool): True if this is an I-frame, False for P-frame.
        sps_id (int): SPS ID (0-15).
        qp (int): Quantization parameter (0-255).
        bit_stream (bytes): The encoded bitstream data for this frame.

    Returns:
        bytes_written (int): Number of bytes written.

    Raises:
        AssertionError: If qp is not in [0, 256).
    """
    written = 0
    nal_type = NalType.NAL_I if is_i_frame else NalType.NAL_P
    flag = (nal_type << 4) + sps_id
    written += write_uchars(fd, [flag])
    assert qp < 256 and qp >= 0
    written += write_uchars(fd, [qp])
    written += write_uint_adaptive(fd, len(bit_stream))
    written += write_bytes(fd, bit_stream)
    return written


def write_picture_pps(fd, pps, bit_stream):
    """
    Write a picture using Picture Parameter Set (PPS) information.

    Similar to write_picture, but uses a PPS dictionary that already
    contains nal_type and sps_id.

    Args:
        fd (file): File descriptor to write to.
        pps (dict): PPS dictionary containing the nal_type, sps_id, and qp.
        bit_stream (bytes): The encoded bitstream data for this frame.

    Returns:
        bytes_written (int): Number of bytes written.
    """
    written = 0
    written += write_uchars(fd, [(pps["nal_type"] << 4) + pps["sps_id"]])
    written += write_uchars(fd, [pps["qp"]])
    written += write_uint_adaptive(fd, len(bit_stream))
    written += write_bytes(fd, bit_stream)
    return written


def read_picture_remaining(fd):
    """
    Read the remaining picture fields from a file descriptor.

    Reads quantization parameter, bitstream length, and bitstream data
    after the initial header (nal_type, sps_id) has been read.

    Args:
        fd (file): File descriptor to read from.

    Returns:
        qp (int): Quantization parameter (0-255)
        bit_stream (bytes): The encoded bitstream data
    """
    qp = read_uchars(fd, 1)[0]
    len_bit_stream = read_uint_adaptive(fd)
    bit_stream = read_bytes(fd, len_bit_stream)
    return qp, bit_stream
