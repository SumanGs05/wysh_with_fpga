import time, argparse
import numpy as np
import serial

ap = argparse.ArgumentParser()
ap.add_argument("port")                       # e.g. /dev/serial0
ap.add_argument("--baud", type=int, default=3000000)
ap.add_argument("--test", action="store_true")
args = ap.parse_args()

FRAME = 17                                    # 0xA5 + 8 channels x 2 bytes
ser = serial.Serial(args.port, args.baud, timeout=0.1)
ser.reset_input_buffer()

buf = bytearray()
chunks = []
prev = None
gaps = 0
badpay = 0
nfr = 0
last = time.time()

def synced(b):
    return len(b) >= FRAME * 4 and all(b[i * FRAME] == 0xA5 for i in range(4))

while True:
    buf += ser.read(4096)

    # find frame alignment
    while len(buf) >= FRAME * 4 and not synced(buf):
        del buf[0]

    if synced(buf):
        n = len(buf) // FRAME
        raw = np.frombuffer(bytes(buf[:n * FRAME]), np.uint8).reshape(n, FRAME)
        badrows = np.nonzero(raw[:, 0] != 0xA5)[0]
        if len(badrows):
            n = int(badrows[0])
            raw = raw[:n]
        if n:
            payload = np.ascontiguousarray(raw[:, 1:])
            if args.test:
                pcm = payload.view('>u2').reshape(n, 8).astype(np.int64)
                c0 = pcm[:, 0]
                if prev is not None:
                    c0 = np.concatenate(([prev], c0))
                prev = int(c0[-1])
                gaps += int(np.sum((np.diff(c0) % 65536) != 1))
                expect = (pcm[:, [0]] + np.arange(8)) % 65536
                badpay += int(np.sum(pcm != expect))
            else:
                chunks.append(payload.view('>i2').reshape(n, 8).astype(np.float64))
            nfr += n
            del buf[:n * FRAME]

    now = time.time()
    if now - last >= 1.0:
        rate = nfr / (now - last)
        if nfr == 0:
            print("no data received - check wiring / FPGA flashed / port", flush=True)
        elif args.test:
            print(f"{rate:8.0f} frames/s | frame gaps: {gaps} | bad payload: {badpay}", flush=True)
        else:
            d = np.vstack(chunks)
            rms = np.sqrt(((d - d.mean(axis=0)) ** 2).mean(axis=0))
            print(f"{rate:8.0f} frames/s | RMS ch0..7:", np.round(rms).astype(int), flush=True)
        chunks = []
        nfr = 0
        last = now