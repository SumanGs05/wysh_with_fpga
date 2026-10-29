import sys, argparse
import numpy as np
import serial

ap = argparse.ArgumentParser()
ap.add_argument("port")
ap.add_argument("--baud", type=int, default=3000000)
ap.add_argument("--ch", default="0")          # e.g. 0  or  0,1,2,3,4,5
ap.add_argument("--gain", type=float, default=8.0)
args = ap.parse_args()

FRAME = 17
sel = [int(x) for x in args.ch.split(",")]
ser = serial.Serial(args.port, args.baud, timeout=0.1)
ser.reset_input_buffer()
out = sys.stdout.buffer

buf = bytearray()
dc = 0.0

def synced(b):
    return len(b) >= FRAME * 4 and all(b[i * FRAME] == 0xA5 for i in range(4))

print(f"channels {sel}, gain {args.gain}", file=sys.stderr)
while True:
    buf += ser.read(4096)
    while len(buf) >= FRAME * 4 and not synced(buf):
        del buf[0]
    if not synced(buf):
        continue
    n = len(buf) // FRAME
    raw = np.frombuffer(bytes(buf[:n * FRAME]), np.uint8).reshape(n, FRAME)
    badrows = np.nonzero(raw[:, 0] != 0xA5)[0]
    if len(badrows):
        n = int(badrows[0]); raw = raw[:n]
    if n == 0:
        continue
    payload = np.ascontiguousarray(raw[:, 1:])
    pcm = payload.view('>i2').reshape(n, 8).astype(np.float32)
    mono = pcm[:, sel].mean(axis=1)
    dc = 0.999 * dc + 0.001 * float(mono.mean())      # slow DC removal
    mono = (mono - dc) * args.gain
    out.write(np.clip(mono, -32768, 32767).astype('<i2').tobytes())
    out.flush()
    del buf[:n * FRAME]