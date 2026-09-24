"""Read bounded records from the Insta360 trailer."""
import struct
from pathlib import Path


MAGIC = b"8db42d694ccc418790edff439fe026bf"


def trailer_record(path, record_id):
    with Path(path).open('rb') as stream:
        size = stream.seek(0, 2)
        if size < 78:
            raise ValueError('INSV file is too short for a trailer')
        stream.seek(size - 72)
        header = stream.read(72)
        if header[-32:] != MAGIC:
            raise ValueError('Invalid INSV magic footer')
        length = struct.unpack_from('<I', header, 32)[0]
        if not 78 <= length <= size:
            raise ValueError('Invalid INSV trailer length')
        start, end = size - length, size - 72
        stream.seek(end - 6)
        fmt, kind, count = struct.unpack('<BBI', stream.read(6))
        entries = {}
        if kind == 0 and fmt == 0:
            if count % 10 or count > length - 78:
                raise ValueError('Malformed INSV trailer directory')
            stream.seek(end - 6 - count)
            for key, flags, data_size, offset in struct.iter_unpack('<BBII', stream.read(count)):
                if key and data_size:
                    entries[key] = (data_size, offset)
        else:
            cursor = end
            while cursor - start >= 6:
                stream.seek(cursor - 6)
                flags, key, data_size = struct.unpack('<BBI', stream.read(6))
                begin = cursor - 6 - data_size
                if begin < start:
                    break
                if key and data_size and key not in entries:
                    entries[key] = (data_size, begin - start)
                cursor = begin
        if record_id not in entries:
            return None
        data_size, offset = entries[record_id]
        if offset < 0 or data_size > end - start or offset + data_size > end - start:
            raise ValueError('INSV trailer record exceeds bounds')
        stream.seek(start + offset)
        return stream.read(data_size)
