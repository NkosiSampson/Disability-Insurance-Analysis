"""
Read the raw records hidden inside the SOA IDEC 2006-2015 pivot files into a
pandas DataFrame. Works for both files:
    idec-2006-15-ct-pivot.xlsb   (claim termination, about 3.3 million rows)
    idec-incidence-table.xlsb    (claim incidence, about 4 million rows)

Why this is needed: pandas / pyxlsb only read the visible summary sheet.
The actual data lives in the pivot cache,
a binary part of the .xlsb file. This script decodes it directly.

Requirements: Python 3.9+, numpy, pandas  (pip install numpy pandas)

Usage:
    python read_idec_pivot.py idec-2006-15-ct-pivot.xlsb            # writes idec_claims.csv
    python read_idec_pivot.py idec-2006-15-ct-pivot.xlsb --pickle   # faster to reload in pandas

Or in your own code:
    from read_idec_pivot import load_idec
    df = load_idec("idec-2006-15-ct-pivot.xlsb")
"""
import struct
import sys
import zipfile

import numpy as np
import pandas as pd

DEF_PATH = "xl/pivotCache/pivotCacheDefinition1.bin"
REC_PATH = "xl/pivotCache/pivotCacheRecords1.bin"


def _read_record(buf, i):
    """Read one BIFF12 record header: returns (type, size, body_start)."""
    t = buf[i]; i += 1
    if t & 0x80:
        t = (t & 0x7F) | ((buf[i] & 0x7F) << 7); i += 1
    size, shift = 0, 0
    for _ in range(4):
        b = buf[i]; i += 1
        size |= (b & 0x7F) << shift; shift += 7
        if not b & 0x80:
            break
    return t, size, i


def _wide_string(b, o):
    n = struct.unpack_from("<I", b, o)[0]
    return b[o + 4:o + 4 + 2 * n].decode("utf-16le", errors="replace"), o + 4 + 2 * n


def read_fields(z):
    """Parse field names and their lookup values (shared items) from the cache definition."""
    d = z.read(DEF_PATH)
    i, fields, cur = 0, [], None
    while i < len(d):
        t, size, i = _read_record(d, i)
        b = d[i:i + size]; i += size
        if t == 0xB7:                      # start of a field: name at offset 20
            cur = {"name": _wide_string(b, 20)[0], "items": [], "calculated": False}
            fields.append(cur)
        elif t == 0x18:                    # one string item
            cur["items"].append(_wide_string(b, 0)[0])
        elif t == 0xBF:                    # block of items (1 = numbers, 2 = strings)
            kind, n, o = b[0], struct.unpack_from("<I", b, 2)[0], 6
            for _ in range(n):
                if kind == 2:
                    s, o = _wide_string(b, o)
                else:
                    s = struct.unpack_from("<d", b, o)[0]; o += 8
                cur["items"].append(s)
        elif t == 0xFD:                    # formula field, not stored in records
            cur["calculated"] = True
    return [f for f in fields if not f["calculated"]]


def load_idec(path, columns=None, chunk_rows=200_000):
    """Return every record in the pivot file as a DataFrame.

    columns: optional list of column names to keep (saves memory).
    """
    with zipfile.ZipFile(path) as z:
        fields = read_fields(z)
        # Each record: 3-byte header, then a 4-byte index for fields that have
        # lookup values, or an 8-byte float for fields that do not.
        dtype = np.dtype([("_hdr", "V3")] + [
            (f["name"], "<u4" if f["items"] else "<f8") for f in fields])
        with z.open(REC_PATH) as fh:
            head = fh.read(7)              # count record: c1 01 04 + uint32
            n_rows = struct.unpack_from("<I", head, 3)[0]
            parts, done = [], 0
            while done < n_rows:
                n = min(chunk_rows, n_rows - done)
                raw = fh.read(n * dtype.itemsize)
                arr = np.frombuffer(raw, dtype=dtype)
                first = np.frombuffer(raw, dtype=np.uint8)[::dtype.itemsize]
                if not np.all(first == 0x21):
                    raise ValueError("Unexpected record layout; this file differs from the one the script was written for.")
                parts.append(arr); done += n
    rec = np.concatenate(parts)

    out = {}
    for f in fields:
        name = f["name"]
        if columns and name not in columns:
            continue
        col = rec[name]
        if f["items"] and isinstance(f["items"][0], str):
            out[name] = pd.Categorical.from_codes(col.astype(np.int32), categories=pd.Index([str(x) for x in f["items"]]))
        elif f["items"]:                   # numeric lookup (e.g. death_count)
            out[name] = np.asarray(f["items"], dtype=float)[col]
        else:
            out[name] = np.asarray(col)
    return pd.DataFrame(out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    df = load_idec(sys.argv[1])
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    if "count_exposed" in df:
        print(f"Check: exposure {df['count_exposed'].sum():,.0f}, deaths {df['death_count'].sum():,.0f}, "
              f"recoveries {df['recovery_count'].sum():,.0f}")
    if "exposure_by_count" in df:
        print(f"Check: policy-years {df['exposure_by_count'].sum():,.0f}, "
              f"new claims {df['disability_count'].sum():,.0f}")
    if "--pickle" in sys.argv:
        df.to_pickle("idec_claims.pkl"); print("Wrote idec_claims.pkl")
    else:
        df.to_csv("idec_claims.csv", index=False); print("Wrote idec_claims.csv")
