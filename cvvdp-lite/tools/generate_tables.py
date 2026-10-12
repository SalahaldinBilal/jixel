#!/usr/bin/env python3
"""Generate the static, sustained-channel CSF data from ColorVideoVDP.

Usage: python3 tools/generate_tables.py /path/to/ColorVideoVDP
The upstream revision and MIT attribution are recorded in NOTICE. This tool
uses only Python's standard library; the Rust library does not parse JSON.
"""
import argparse
import json
import math
from pathlib import Path
import struct


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'src/tables.rs')
    args = parser.parse_args()
    lut = json.loads((args.upstream/'pycvvdp/vvdp_data/csf_lut_weber_fixed_size.json').read_text())
    assert len(lut['L_bkg']) == len(lut['rho']) == 32
    assert all(len(row) == 32 for ch in range(1, 4) for row in lut[f'o0_c{ch}'])
    lines = ['// Generated from ColorVideoVDP 0.5.7; see NOTICE and tools/generate_tables.py.',
             '#![allow(clippy::excessive_precision)]']
    for name, values in [('LOG_LUMINANCE', lut['L_bkg']), ('LOG_FREQUENCY', lut['rho'])]:
        lines.append(f'pub(crate) const {name}: [f32; 32] = ['+
                     ','.join(repr(f32(math.log10(v))) for v in values)+'];')
    lines.append('pub(crate) const LOG_SENSITIVITY: [[[f32; 32]; 32]; 3] = [')
    for channel in range(1, 4):
        lines.append('[')
        lines.extend('['+','.join(repr(f32(v)) for v in row)+'],' for row in lut[f'o0_c{channel}'])
        lines.append('],')
    lines.append('];')
    args.output.write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
