# Read time-division photometry recorded by a Neuropixels OneBox (SpikeGLX obx stream).
# Licenced under the GNU General Public License v3, like the rest of pyPhotometry.

"""Time-division photometry from a Neuropixels OneBox recording (SpikeGLX obx stream).

pyPhotometry's pulsed modes read each photoreceiver twice per LED pulse on the
pyboard: a baseline over the 250 us before the LED turns on, then the LED-on signal
over 250 us starting 300 us after it turns on (uPy/photometry_upy.py, pulsed_ISR), and
store LED-on and baseline separately (.ppd v1.1); data_import.import_ppd subtracts them.

Here the OneBox records the photoreceivers continuously at ~30.3 kHz together with the
LED gate signals, and this module applies the same windows to the recorded samples,
using the gate edges the OneBox saw. Output is a dict shaped like import_ppd's, so
data_import.preprocess_data runs on it unchanged, or a .ppd file import_ppd can open.
Sample times come from the OneBox clock, and the SpikeGLX sync wave (SY bit 6) maps
them onto any probe stream of the same run.
"""
import json
import datetime
import numpy as np

BASE_S = 250e-6  # pyPhotometry: 64 ADC reads at 256 kHz just before LED on
WAIT_S = 300e-6  # pyb.udelay(300) after LED on
READ_S = 250e-6  # 64 ADC reads at 256 kHz, then LED off


def window_means(sig, start, stop):
    """Mean of sig over [start, stop) for float sample bounds (whole samples whose index
    lies in the interval), vectorised with a cumulative sum. Empty window -> NaN."""
    cs = np.concatenate([[0.0], np.cumsum(sig, dtype=np.float64)])
    a = np.clip(np.ceil(start).astype(np.int64), 0, sig.size)
    b = np.clip(np.ceil(stop).astype(np.int64), 0, sig.size)
    n = b - a
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[b] - cs[a]) / n, np.nan), n


def demodulate(sig, on, off, fs, base_s=BASE_S, wait_s=WAIT_S, read_s=READ_S):
    """pyPhotometry pulsed-mode windows on a continuously sampled photoreceiver.

    sig: photoreceiver volts (1-D); on/off: LED gate edges in samples of sig (float ok);
    fs: sample rate of sig. Returns (led_on, baseline, t_read, n_on, n_base), where
    t_read is the centre of the LED-on window in samples.
    """
    on = np.asarray(on, float)
    off = np.asarray(off, float)
    base, n_base = window_means(sig, on - base_s * fs, on)
    r0 = on + wait_s * fs
    r1 = np.minimum(on + (wait_s + read_s) * fs, off)
    led_on, n_on = window_means(sig, r0, r1)
    return led_on, base, 0.5 * (np.ceil(r0) + np.ceil(r1) - 1), n_on, n_base


def gate_edges(bits):
    """(on, off) sample indices of a 0/1 gate line; on is the first high sample."""
    b = np.asarray(bits).astype(np.int8)
    d = np.diff(b)
    on = np.flatnonzero(d == 1) + 1
    off = np.flatnonzero(d == -1) + 1
    if off.size and on.size and off[0] < on[0]:
        off = off[1:]
    m = min(on.size, off.size)
    return on[:m], off[:m]


def sequence(gates):
    """Merge per-LED (on, off) arrays into one time-ordered list of pulses and keep only
    whole frames (each LED once, in the first frame's order). Returns
    (on, off, led, frame_order); a dropped or extra pulse ends the frame it is in."""
    on = np.concatenate([g[0] for g in gates])
    off = np.concatenate([g[1] for g in gates])
    led = np.concatenate([np.full(g[0].size, j) for j, g in enumerate(gates)])
    o = np.argsort(on, kind="stable")
    on, off, led = on[o], off[o], led[o]
    n = len(gates)
    order = led[:n]
    keep = np.zeros(on.size, bool)
    i = 0
    while i + n <= on.size:
        if np.array_equal(led[i : i + n], order):
            keep[i : i + n] = True
            i += n
        else:
            i += 1
    return on[keep], off[keep], led[keep], order


def to_ppd_dict(led_on, base, led, t_read, fs_obx, mode="3EX_2EM_pulsed", digital=None,
                analog_of_led=None, extra=None):
    """Arrange demodulated pulses as import_ppd's dict: analog_k = LED k's signal
    (baseline subtracted), analog_k_raw_LED_on / _raw_baseline, time (ms from the first
    frame). led holds 0-based LED ids in frame order; analog_of_led maps LED id -> analog
    number (default LED k -> analog k+1, which is pyPhotometry's 3EX_2EM_pulsed: LED1 on
    analog 1, LED2 on analog 2, LED3 on analog 1 stored as signal 3)."""
    n_led = int(led.max()) + 1
    n_frames = led.size // n_led
    d = {"mode": mode, "n_analog_signals": n_led}
    t_frames = t_read[led == led[0]][:n_frames]
    d["time"] = (t_frames - t_frames[0]) * 1000.0 / fs_obx
    d["sampling_rate"] = float((n_frames - 1) / ((t_frames[-1] - t_frames[0]) / fs_obx))
    d["obx_sample"] = t_frames
    for j in range(n_led):
        k = (analog_of_led or {}).get(j, j + 1)
        lo = led_on[led == j][:n_frames]
        bl = base[led == j][:n_frames]
        d[f"analog_{k}"] = lo - bl
        d[f"analog_{k}_raw_LED_on"] = lo
        d[f"analog_{k}_raw_baseline"] = bl
        d[f"obx_sample_{k}"] = t_read[led == j][:n_frames]
    if digital is not None:
        d["digital_1"] = digital
    d.update(extra or {})
    return d


def write_ppd(path, d, volts_per_division, subject_ID="onebox", version="1.1.0",
              digital=None):
    """Write a pyPhotometry v1.1 .ppd (pulsed mode: LED-on and baseline stored
    separately) that tools/data_import.import_ppd opens. Analog values are 15-bit
    unsigned: volts/volts_per_division, clipped to [0, 32767]; pick volts_per_division =
    the OneBox range / 32768 so a non-negative OneBox sample maps exactly."""
    n_a = d["n_analog_signals"]
    n_f = d["time"].size
    words = np.zeros((n_f, n_a, 2), np.uint16)
    for k in range(1, n_a + 1):
        for i, key in enumerate((f"analog_{k}_raw_LED_on", f"analog_{k}_raw_baseline")):
            v = np.clip(np.round(d[key] / volts_per_division), 0, 32767)
            words[:, k - 1, i] = v.astype(np.uint16) << 1
    if digital is not None:
        words[:, 0, 0] |= np.asarray(digital[:n_f], np.uint16) & 1
    now = datetime.datetime.now().isoformat(timespec="seconds")
    header = {
        "subject_ID": subject_ID, "date_time": now, "end_time": now, "mode": d["mode"],
        "sampling_rate": d["sampling_rate"], "volts_per_division": [volts_per_division] * 2,
        "LED_current": [0, 0], "version": version, "n_analog_signals": n_a,
        "n_digital_signals": 1, "ADC_max_value": 32767,
        "source": "OneBox obx stream via onebox_import.py",
    }
    hb = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(len(hb).to_bytes(2, "little"))
        f.write(hb)
        f.write(words.astype("<u2").tobytes())


def pair_sync_edges(t_a, t_b, max_dt=0.5):
    """Pair sync edges of two streams given times (s) on a rough common clock (e.g.
    (firstSample + index) / rate since the shared acquisition start). Returns index arrays."""
    j = np.searchsorted(t_b, t_a)
    j = np.clip(j, 1, t_b.size - 1)
    jj = np.where(np.abs(t_b[j - 1] - t_a) < np.abs(t_b[j] - t_a), j - 1, j)
    ok = np.abs(t_b[jj] - t_a) < max_dt
    ia = np.flatnonzero(ok)
    return ia, jj[ok]


def fit_map(x, y):
    """Least-squares y = a + b x; returns (a, b, residuals)."""
    A = np.vstack([np.ones_like(x), x]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return a, b, y - (a + b * x)
