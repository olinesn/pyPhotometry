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
import argparse
import datetime
import hashlib
import json
import os
import socket
import numpy as np

TOOL_VERSION = "onebox-0.3"

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


def demodulate(sig, on, off, fs, base_s=BASE_S, wait_s=WAIT_S, read_s=READ_S, interp=True):
    """pyPhotometry pulsed-mode windows on a continuously sampled photoreceiver.

    sig: photoreceiver volts (1-D); on/off: LED gate edges in samples of sig (float ok);
    fs: sample rate of sig. Returns (led_on, baseline, t_read, n_on, n_base), where
    t_read is the centre of the LED-on window in samples.

    interp=True (default) reads each window at the same times relative to its gate edge
    (round(window * fs) evenly spaced points, linearly interpolated between samples).
    Without it, a window takes the whole samples inside it, and because the LED timing
    and the OneBox sampling drift against each other, the window's position relative to
    the edge wanders by up to one sample: on a photoreceiver still settling 300 us into
    the pulse that is a slow ripple of ~1 % of the signal (measured on simulated data).
    """
    on = np.asarray(on, float)
    off = np.asarray(off, float)
    r0 = on + wait_s * fs
    r1 = np.minimum(on + (wait_s + read_s) * fs, off)
    if not interp:
        base, n_base = window_means(sig, on - base_s * fs, on)
        led_on, n_on = window_means(sig, r0, r1)
        return led_on, base, 0.5 * (np.ceil(r0) + np.ceil(r1) - 1), n_on, n_base

    idx = np.arange(sig.size)

    def grid_mean(start, length):
        n = max(1, int(round(np.median(length))))
        frac = (np.arange(n) + 0.5) / n
        pos = start[:, None] + frac[None, :] * length[:, None]
        return np.interp(pos, idx, sig).mean(1), np.full(start.size, n)

    # One-sample guards: a point interpolated within one sample of an LED edge would mix
    # in samples from the other side of it (the light rising or falling), by an amount
    # that depends on the sampling phase. pyPhotometry reads its baseline right up to the
    # edge; here the baseline ends one sample (33 us) earlier and the read ends one
    # sample before the LED turns off.
    base, n_base = grid_mean(on - 1 - base_s * fs, np.full(on.size, base_s * fs))
    r1 = np.minimum(r1, off - 1)
    led_on, n_on = grid_mean(r0, r1 - r0)
    return led_on, base, 0.5 * (r0 + r1), n_on, n_base


def refine_edges(edges, seg=2000):
    """Sub-sample gate edges. A gate recorded by the OneBox switches at the first sample
    at or after the true edge, so it is late by 0-1 sample. pyPhotometry's LED pulses come
    from one timer (all LEDs, every slot equally spaced), so the true edges lie on a
    straight line against the pulse number; fit it piecewise (seg pulses, to follow
    slow clock drift) and subtract the half sample. Pass the merged, time-ordered edges
    of all LEDs (sequence() output). Returns (refined, residual sd in samples)."""
    e = np.asarray(edges, float)
    j = np.arange(e.size, dtype=float)
    out = np.empty_like(e)
    res = []
    for s0 in range(0, e.size, seg):
        s1 = min(e.size, s0 + seg)
        if s1 - s0 < 10:  # too short to fit: extend the previous segment's line
            s0 = max(0, s1 - seg)
        A = np.vstack([np.ones(s1 - s0), j[s0:s1]]).T
        c, *_ = np.linalg.lstsq(A, e[s0:s1], rcond=None)
        fit = A @ c
        out[s0:s1] = fit - 0.5
        res.append(e[s0:s1] - fit)
    return out, float(np.concatenate(res).std())


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
    d["obx_sample_rate"] = float(fs_obx)
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


def write_ppd(path, d, volts_per_division, subject_id="", session_id="", fibers=None,
              time_reference="obx0", probe_clock=None, version="1.1.0", digital=None,
              extra_header=None):
    """Write a pyPhotometry v1.1 .ppd (pulsed mode: LED-on and baseline stored
    separately) that tools/data_import.import_ppd opens. Analog values are 15-bit
    unsigned: volts/volts_per_division, clipped to [0, 32767]; pick volts_per_division =
    the OneBox range / 32768 so a non-negative OneBox sample maps exactly.

    Header additions beyond pyPhotometry's own (import_ppd copies them into its dict):
      session_id, subject_id  - the lab's session and subject IDs (DataJoint keys)
      fibers                  - [{channel, fiber_id, region, hemisphere, indicator}]
      time_reference          - the clock the sample times are on ("obx0": the OneBox's)
      obx_first_sample, obx_sample_rate - frame 0's read time on that clock (samples of
                                the .obx.bin) and its rate; exact per-frame times are in
                                the <name>.obx_samples.npy sidecar
      probe_clock             - once mapped: {stream, slope, offset_s, residual_sd_s,
                                residual_max_s, n_edges}, t_probe = offset_s + slope * t_obx
                                (seconds from each file's first sample)
    """
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
        "subject_ID": subject_id, "date_time": now, "end_time": now, "mode": d["mode"],
        "sampling_rate": d["sampling_rate"], "volts_per_division": [volts_per_division] * 2,
        "LED_current": [0, 0], "version": version, "n_analog_signals": n_a,
        "n_digital_signals": 1, "ADC_max_value": 32767,
        "source": "OneBox obx stream via onebox_import.py",
        "session_id": session_id, "subject_id": subject_id, "fibers": fibers or [],
        "time_reference": time_reference,
        "obx_first_sample": float(d["obx_sample"][0]) if "obx_sample" in d else None,
        "obx_sample_rate": d.get("obx_sample_rate"),
    }
    if probe_clock:
        header["probe_clock"] = probe_clock
    header.update(extra_header or {})
    hb = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(len(hb).to_bytes(2, "little"))
        f.write(hb)
        f.write(words.astype("<u2").tobytes())
    if "obx_sample" in d:
        np.save(os.path.splitext(path)[0] + ".obx_samples.npy", np.asarray(d["obx_sample"], float))


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def write_analysis_json(path, session_id, inputs, outputs, params, status="draft",
                        probe_clock=None, root=None):
    """Write the lab's analysis.json sidecar (schema lab-analysis/1) beside the outputs.
    inputs/outputs: [(path, runRef or None)]; paths are stored relative to root (default:
    the folder of `path`), with their SHA-256."""
    root = root or os.path.dirname(os.path.abspath(path))

    def ent(p, ref):
        e = {"path": os.path.relpath(os.path.abspath(p), root).replace(os.sep, "/"),
             "sha256": sha256_file(p)}
        if ref:
            e["runRef"] = ref
        return e

    doc = {
        "schema": "lab-analysis/1", "sessionId": session_id,
        "analysis": "photometry-demodulation", "tool": "pyPhotometry onebox_import",
        "toolVersion": TOOL_VERSION,
        "inputs": [ent(p, r) for p, r in inputs],
        "outputs": [ent(p, r) for p, r in outputs],
        "params": params, "status": status,
        "host": socket.gethostname(), "ended": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if probe_clock:
        doc["clockMap"] = probe_clock
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    return doc


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


# ----------------------------------------------------------------------------------
# SpikeGLX files -> .ppd + analysis.json (no human input beyond the rig's channel map)
# ----------------------------------------------------------------------------------


def read_spikeglx(bin_path):
    """(meta dict, int16 memmap [samples, channels]) of a SpikeGLX .bin."""
    meta = {}
    with open(os.path.splitext(bin_path)[0] + ".meta") as f:
        for line in f:
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                meta[k.lstrip("~")] = v
    n_ch = int(meta["nSavedChans"])
    data = np.memmap(bin_path, dtype="<i2", mode="r")
    return meta, data[: data.size // n_ch * n_ch].reshape(-1, n_ch)


def gate_edges_analog(v, frac=0.5):
    """Gate (on, off) from an analog copy of an LED drive or TTL: threshold half way
    between its off level (5th percentile) and on level (99.9th percentile). Works for
    pyPhotometry v2.0's LED control outputs, whose on level scales with LED current."""
    lo, hi = np.percentile(v, [5, 99.9])
    return gate_edges(v > lo + frac * (hi - lo))


def sync_map(obx_meta, obx, imec_bin):
    """Fit t_probe = offset_s + slope * t_obx from SY bit 6 (seconds from each file's
    first sample), pairing edges by time since the shared acquisition start."""
    meta_i, arr_i = read_spikeglx(imec_bin)
    fs_o, fs_i = float(obx_meta["obSampRate"]), float(meta_i["imSampRate"])
    e_o = np.flatnonzero(np.diff((obx[:, -1] >> 6) & 1) == 1) + 1
    e_i = np.flatnonzero(np.diff((arr_i[:, -1] >> 6) & 1) == 1) + 1
    t_o = (int(obx_meta.get("firstSample", 0)) + e_o) / fs_o
    t_i = (int(meta_i.get("firstSample", 0)) + e_i) / fs_i
    io, ii = pair_sync_edges(t_o, t_i)
    a, b, r = fit_map(e_o[io] / fs_o, e_i[ii] / fs_i)
    name = os.path.basename(imec_bin).split(".")[-3] if imec_bin.count(".") >= 3 else "imec"
    return {"stream": name, "slope": float(b), "offset_s": float(a),
            "residual_sd_s": float(r.std()), "residual_max_s": float(np.abs(r).max()),
            "n_edges": int(io.size)}


def convert(obx_bin, out_ppd, pr_xa, gate_xa, led_pr=(0, 1, 0), imec_bin=None,
            session_id="", subject_id="", fibers=None, status="draft", board_ppd=None,
            mode="3EX_2EM_pulsed", refine=True):
    """OneBox recording -> .ppd (+ .obx_samples.npy) + analysis.json.
    pr_xa: XA positions of the photoreceivers; gate_xa: XA position of each LED's gate,
    in LED order; led_pr: photoreceiver index each LED is read on (3EX_2EM_pulsed:
    LED1 -> PR1, LED2 -> PR2, LED3 -> PR1). board_ppd: the pyPhotometry board's own file
    of the same session, if any, listed as an input."""
    os.makedirs(os.path.dirname(os.path.abspath(out_ppd)), exist_ok=True)
    meta, arr = read_spikeglx(obx_bin)
    fs = float(meta["obSampRate"])
    vpb = float(meta["obAiRangeMax"]) / float(meta.get("obMaxInt", 32768))
    gates = [gate_edges_analog(arr[:, c].astype(np.float32) * vpb) for c in gate_xa]
    on, off, led, order = sequence(gates)
    on_r, on_sd = refine_edges(on)
    if not refine:  # use the recorded edges as they are (minus the mean half sample)
        on_r = on - 0.5
    off_r = on_r + np.median(off - on)  # pulses have one length (the firmware's)
    led_on = np.empty(on.size)
    base = np.empty(on.size)
    t_read = np.empty(on.size)
    for j, p in enumerate(led_pr):
        m = led == j
        sig = arr[:, pr_xa[p]].astype(np.float32) * vpb
        led_on[m], base[m], t_read[m], *_ = demodulate(sig, on_r[m], off_r[m], fs)
    d = to_ppd_dict(led_on, base, led, t_read, fs, mode=mode)
    clock = sync_map(meta, arr, imec_bin) if imec_bin else None
    obx_name = os.path.basename(obx_bin).split(".")[-3]
    write_ppd(out_ppd, d, vpb, subject_id=subject_id, session_id=session_id, fibers=fibers,
              time_reference=obx_name, probe_clock=clock)
    params = {"base_s": BASE_S, "wait_s": WAIT_S, "read_s": READ_S, "mode": mode,
              "gate_threshold": "half way between 5th and 99.9th percentile",
              "pr_xa": list(pr_xa), "gate_xa": list(gate_xa), "led_pr": list(led_pr),
              "obx_sample_rate": fs, "volts_per_division": vpb,
              "window_reads": "interpolated at fixed times after each gate edge, 1-sample guards",
              "gate_edges": "refined on the LED timer grid" if refine else "as recorded - 0.5 sample",
              "gate_edge_fit_resid_samples": on_sd,
              "frames": int(d["time"].size), "led_order": [int(x) for x in order]}
    inputs = [(obx_bin, obx_name)]
    if board_ppd:
        inputs.append((board_ppd, "pyphotometry"))
    if imec_bin:
        inputs.append((os.path.splitext(imec_bin)[0] + ".meta", clock["stream"]))
    outputs = [(out_ppd, None), (os.path.splitext(out_ppd)[0] + ".obx_samples.npy", None)]
    return write_analysis_json(os.path.join(os.path.dirname(os.path.abspath(out_ppd)), "analysis.json"),
                               session_id, inputs, outputs, params, status=status,
                               probe_clock=clock)


def main(argv=None):
    ap = argparse.ArgumentParser(description="OneBox photometry -> pyPhotometry .ppd + analysis.json")
    ap.add_argument("obx_bin", help="SpikeGLX <run>.obx0.obx.bin")
    ap.add_argument("out_ppd")
    ap.add_argument("--pr", type=int, nargs="+", required=True, help="XA positions of the photoreceivers")
    ap.add_argument("--gates", type=int, nargs="+", required=True, help="XA position of each LED's gate, LED order")
    ap.add_argument("--led-pr", type=int, nargs="+", default=[0, 1, 0], help="photoreceiver each LED is read on")
    ap.add_argument("--imec", help="a probe .ap.bin of the same run, to map times onto its clock")
    ap.add_argument("--session-id", default="")
    ap.add_argument("--subject-id", default="")
    ap.add_argument("--fibers", help="JSON file: [{channel, fiber_id, region, hemisphere, indicator}]")
    ap.add_argument("--board-ppd", help="the pyPhotometry board's own .ppd of this session")
    ap.add_argument("--status", default="draft", choices=["draft", "final", "superseded"])
    ap.add_argument("--no-refine", action="store_true",
                    help="use recorded gate edges as they are (if the LED timing is not regular: "
                         "analysis.json's gate_edge_fit_resid_samples well above 0.29)")
    a = ap.parse_args(argv)
    fibers = json.load(open(a.fibers)) if a.fibers else None
    doc = convert(a.obx_bin, a.out_ppd, a.pr, a.gates, a.led_pr, a.imec, a.session_id,
                  a.subject_id, fibers, a.status, a.board_ppd, refine=not a.no_refine)
    print(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
