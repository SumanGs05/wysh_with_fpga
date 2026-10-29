#!/usr/bin/env python3
"""
WYSH beamformer - Raspberry Pi side
FPGA (Tang Nano 9K) --UART 3 Mbaud--> Pi --> MVDR / delay-and-sum --> headphones

Stream format sent by the FPGA (one frame per audio sample):
    0xA5 | ch0_hi ch0_lo | ch1_hi ch1_lo | ... | ch7_hi ch7_lo     (17 bytes, int16 big-endian)
    ch0..ch5 = mics 0..5 (D0..D2),  ch6 = empty slot,  ch7 = centre mic (D3)

Usage:
    python3 wysh_beamformer.py                 # uses SERIAL_PORT below
    python3 wysh_beamformer.py /dev/ttyUSB0    # override the port

Needs only numpy + pyserial. Audio goes out through `aplay` (no PyAudio).

Controls while running:
    <number>   steer the beam, degrees  (0 = straight ahead, +90 = right, -90 = left, +-180 = behind)
    m          toggle MVDR <-> delay-and-sum
    g <x>      output gain, e.g.  g 5
    q          quit
"""
import sys
import time
import threading
import subprocess
from collections import deque

import numpy as np
import serial

# =============================== CONFIG ===============================
SERIAL_PORT = '/dev/serial0'      # FPGA UART on Pi pin 10 (GPIO15). USB device would be /dev/ttyUSB0 or /dev/ttyACM0
SERIAL_BAUD = 3000000

SAMPLE_RATE = 27e6 / 26 / 64      # ~16226 Hz, fixed by the FPGA clock divider
ALSA_DEVICE = 'plughw:Headphones' # Pi 3.5 mm jack (see `aplay -l`)

# --- which FPGA channels to use and where those mics physically sit -------------
# 6-mic ring: ch0..ch5 = mics 0..5.  Coordinates in metres: x = right, y = straight ahead.
# Azimuth 0 = straight ahead (+y), +90 = right (+x), -90 = left.
#
# !! Two things below are ASSUMPTIONS until you check them on your board:
#    RING_RADIUS_M    - measure centre-of-board to a ring mic (or half the distance between opposite mics)
#    RING_ANGLES_DEG  - angle of each channel around the ring, 0 = top of the board, clockwise.
#                       Default assumes ch0..ch5 run around the ring in order, 60 deg apart.
#                       Tap each mic while running wysh_rx.py and edit this list to match.
RING_RADIUS_M = 0.046
RING_ANGLES_DEG = [0, 60, 120, 180, 240, 300]

MIC_CHANNELS = [0, 1, 2, 3, 4, 5]
_ang = np.radians(RING_ANGLES_DEG)
MIC_POSITIONS = RING_RADIUS_M * np.stack([np.sin(_ang), np.cos(_ang)], axis=1)

# Add the centre mic (ch7) at the origin by using:
#   MIC_CHANNELS = [0, 1, 2, 3, 4, 5, 7]
#   MIC_POSITIONS = np.vstack([MIC_POSITIONS, [0.0, 0.0]])

AZIMUTH_SIGN = 1.0     # set to -1.0 if the beam turns out mirrored left <-> right

# --- DSP ----------------------------------------------------------------------
NFFT = 256             # 63 Hz bins, 16 ms window
HOP = NFFT // 2        # sqrt-Hann analysis+synthesis at 50 % overlap = perfect reconstruction
FREQ_MIN = 300.0       # MVDR is applied in this band; outside it plain delay-and-sum
FREQ_MAX = 4000.0
HIGH_PASS_HZ = 100.0   # bins below this are zeroed (removes DC and rumble)
ALPHA = 0.98           # covariance smoothing per frame (~127 frames/s -> ~0.4 s memory)
LOADING = 0.05         # diagonal loading, relative to signal power (higher = safer, closer to DAS)
OUTPUT_GAIN = 7.0
MAX_BUFFER_S = 0.3     # if the Pi falls behind by more than this, old audio is dropped

SPEED_OF_SOUND = 343.0

# --- stream constants (must match the FPGA) ------------------------------------
SYNC = 0xA5
N_STREAM_CH = 8
FRAME = 1 + 2 * N_STREAM_CH   # 17 bytes

# =============================== STATE ================================
running = True
azimuth = 0.0
mode = 'mvdr'
gain = OUTPUT_GAIN
stats = {'frames': 0, 'dropped': 0, 'resync': 0}


# ============================ FRAME PARSER ============================
def extract_frames(buf):
    """Pull complete, aligned frames out of bytearray `buf` (consumed in place).
    Returns int16 array (n, 8) or None."""
    while len(buf) >= 4 * FRAME and not (buf[0] == SYNC and buf[FRAME] == SYNC and
                                         buf[2 * FRAME] == SYNC and buf[3 * FRAME] == SYNC):
        stats['resync'] += 1
        nxt = buf.find(b'\xa5', 1)
        if nxt < 0:
            buf.clear()
        else:
            del buf[:nxt]
    if len(buf) < 4 * FRAME:
        return None
    n = len(buf) // FRAME
    raw = np.frombuffer(bytes(buf[:n * FRAME]), np.uint8).reshape(n, FRAME)
    bad = np.flatnonzero(raw[:, 0] != SYNC)
    if len(bad):
        n = int(bad[0])
        raw = raw[:n]
    if n == 0:
        return None
    pcm = np.ascontiguousarray(raw[:, 1:]).view('>i2').reshape(n, N_STREAM_CH)
    del buf[:n * FRAME]
    return pcm


# ============================ FRAME QUEUE =============================
class FrameQueue:
    """Thread-safe queue of sample blocks; drops the oldest audio if the consumer lags."""
    def __init__(self, max_frames):
        self.q = deque()
        self.n = 0
        self.max = max_frames
        self.cv = threading.Condition()

    def put(self, arr):
        with self.cv:
            self.q.append(arr)
            self.n += len(arr)
            while self.n > self.max and len(self.q) > 1:
                old = self.q.popleft()
                self.n -= len(old)
                stats['dropped'] += len(old)
            self.cv.notify()

    def get(self, timeout=0.2):
        with self.cv:
            if not self.q:
                self.cv.wait(timeout)
            if not self.q:
                return None
            arr = self.q.popleft()
            self.n -= len(arr)
            return arr


# ============================= BEAMFORMER =============================
class Beamformer:
    def __init__(self, positions):
        self.pos = np.asarray(positions, dtype=float)
        self.M = len(self.pos)
        self.win = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(NFFT) / NFFT))
        self.freqs = np.fft.rfftfreq(NFFT, 1.0 / SAMPLE_RATE)
        self.F = len(self.freqs)
        self.band = np.flatnonzero((self.freqs >= FREQ_MIN) & (self.freqs <= FREQ_MAX))
        self.hp = self.freqs < HIGH_PASS_HZ
        self.R = np.tile(np.eye(self.M) * 1e-6, (self.F, 1, 1)).astype(complex)
        self.ola = np.zeros(NFFT)
        self._az = None
        self._a = None

    def steering(self, az_deg):
        """a[f, m]: phase of a plane wave from azimuth az at mic m (mics nearer the source hear it earlier)."""
        if az_deg != self._az:
            r = np.radians(az_deg)
            u = np.array([np.sin(r), np.cos(r)])
            tau = self.pos @ u / SPEED_OF_SOUND
            self._a = np.exp(1j * 2 * np.pi * self.freqs[:, None] * tau[None, :])
            self._az = az_deg
        return self._a

    def process(self, frame, az_deg, mode):
        """frame: (NFFT, M) float -> HOP output samples."""
        X = np.fft.rfft(frame * self.win[:, None], axis=0)                 # (F, M)
        self.R = ALPHA * self.R + (1 - ALPHA) * (X[:, :, None] * X[:, None, :].conj())

        a = self.steering(az_deg)
        w = a / self.M                                                      # delay-and-sum weights
        if mode == 'mvdr' and self.M > 1 and len(self.band):
            idx = self.band
            Rb = self.R[idx]
            tr = np.trace(Rb, axis1=1, axis2=2).real / self.M
            Rb = Rb + (LOADING * tr + 1e-10)[:, None, None] * np.eye(self.M)
            ab = a[idx]
            Ria = np.linalg.solve(Rb, ab[..., None])[..., 0]                # R^-1 a
            den = np.einsum('fm,fm->f', ab.conj(), Ria).real
            w = w.copy()
            w[idx] = Ria / den[:, None]                                     # distortionless in look direction

        Y = np.einsum('fm,fm->f', w.conj(), X)
        Y[self.hp] = 0.0
        y = np.fft.irfft(Y, NFFT) * self.win
        self.ola += y
        out = self.ola[:HOP].copy()
        self.ola = np.concatenate([self.ola[HOP:], np.zeros(HOP)])
        return out


# ================================ THREADS =============================
def reader_loop(ser, fq):
    buf = bytearray()
    while running:
        try:
            data = ser.read(max(1, ser.in_waiting))
        except Exception as e:
            print(f"\nserial error: {e}")
            time.sleep(0.5)
            continue
        if not data:
            continue
        buf += data
        pcm = extract_frames(buf)
        if pcm is None:
            continue
        sel = pcm[:, MIC_CHANNELS].astype(np.float64) / 32768.0
        stats['frames'] += len(sel)
        fq.put(sel)


def dsp_loop(fq, player):
    global running
    bf = Beamformer(MIC_POSITIONS)
    in_buf = np.zeros((0, bf.M))
    while running:
        arr = fq.get()
        if arr is None:
            continue
        in_buf = np.concatenate([in_buf, arr])
        outs = []
        while len(in_buf) >= NFFT:
            outs.append(bf.process(in_buf[:NFFT], AZIMUTH_SIGN * azimuth, mode))
            in_buf = in_buf[HOP:]
        if outs:
            y = np.clip(np.concatenate(outs) * gain, -1.0, 1.0)
            try:
                player.stdin.write((y * 32767).astype('<i2').tobytes())
                player.stdin.flush()
            except (BrokenPipeError, ValueError):
                print("\naplay closed - stopping (check ALSA_DEVICE / `aplay -l`)")
                running = False


def stats_loop():
    last_t, last_n = time.time(), 0
    while running:
        time.sleep(5.0)
        now = time.time()
        n = stats['frames']
        print(f"[stats] {(n - last_n) / (now - last_t):7.0f} frames/s | dropped {stats['dropped']} | "
              f"resync {stats['resync']} | az {azimuth:+.0f} | {mode} | gain {gain:g}")
        last_t, last_n = now, n


# ================================ MAIN ================================
def main():
    global running, azimuth, mode, gain
    port = sys.argv[1] if len(sys.argv) > 1 else SERIAL_PORT

    d = [np.linalg.norm(MIC_POSITIONS[i] - MIC_POSITIONS[j])
         for i in range(len(MIC_POSITIONS)) for j in range(i + 1, len(MIC_POSITIONS))]
    print("=" * 66)
    print("WYSH beamformer")
    print(f"  serial   : {port} @ {SERIAL_BAUD}")
    print(f"  channels : {MIC_CHANNELS}  ({len(MIC_CHANNELS)} mics)")
    if d:
        print(f"  spacing  : {min(d) * 1000:.1f} mm min  -> spatial aliasing above ~{SPEED_OF_SOUND / (2 * min(d)):.0f} Hz")
    print(f"  audio out: {ALSA_DEVICE}")
    print("=" * 66)

    try:
        ser = serial.Serial(port, SERIAL_BAUD, timeout=0.1)
        ser.reset_input_buffer()
    except Exception as e:
        print(f"Cannot open {port}: {e}")
        print("Check: ls -l /dev/serial0  |  python3 -m serial.tools.list_ports -v")
        sys.exit(1)

    player = subprocess.Popen(
        ['aplay', '-q', '-D', ALSA_DEVICE, '-t', 'raw', '-f', 'S16_LE',
         '-r', str(int(round(SAMPLE_RATE))), '-c', '1', '-B', '200000'],
        stdin=subprocess.PIPE)

    fq = FrameQueue(int(MAX_BUFFER_S * SAMPLE_RATE))
    for fn, args in ((reader_loop, (ser, fq)), (dsp_loop, (fq, player)), (stats_loop, ())):
        threading.Thread(target=fn, args=args, daemon=True).start()

        print("""
Controls while running:
    <number>   steer the beam, degrees (0 = ahead, +90 = right, -90 = left, +-180 = behind)
    m          toggle MVDR <-> delay-and-sum
    g <x>      output gain, e.g.  g 5
    q          quit
""")
    while running:
        try:
            cmd = input("beam> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            break
        if not cmd:
            continue
        if cmd == 'q':
            break
        if cmd == 'm':
            mode = 'das' if mode == 'mvdr' else 'mvdr'
            print(f"mode: {mode}")
        elif cmd.startswith('g'):
            try:
                gain = float(cmd[1:])
                print(f"gain: {gain:g}")
            except ValueError:
                print("usage: g 5")
        else:
            try:
                a = float(cmd)
                if -180 <= a <= 180:
                    azimuth = a
                    print(f"beam -> {a:+.0f} deg")
                else:
                    print("angle must be -180..180")
            except ValueError:
                print("unknown command")

    running = False
    time.sleep(0.3)
    try:
        player.stdin.close()
    except Exception:
        pass
    player.terminate()
    ser.close()
    print("done")


if __name__ == '__main__':
    main()
