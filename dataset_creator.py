#!/usr/bin/env python3
import os
import argparse
import numpy as np
import cv2
import tensorflow as tf

# ----------------------------
# Helpers for TFRecords fields
# ----------------------------
def _bytes_feature(b): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))
def _int64_feature(v): return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))

# ----------------------------
# 1) Frames -> PNGs + frame_times
# ----------------------------
def extract_frames_to_png(video_path, out_dir, cam='left'):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 1:
        raise RuntimeError("Video has <= 1 frame; need at least 2.")

    times = np.array([i / float(fps) for i in range(n)], dtype=np.float64)

    first_gray = None
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {i} from {video_path}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if first_gray is None:
            first_gray = gray.copy()
        cv2.imwrite(os.path.join(out_dir, f"{cam}_image{i:05d}.png"), gray)

    cap.release()
    H, W = first_gray.shape[:2]
    return times, H, W

# ----------------------------
# 2) Load v2e events TXT (t, x, y, p)
# ----------------------------
def load_events_txt(path):
    # Accepts whitespace-delimited columns: t(s) float, x int, y int, p int (+1/-1)
    ev = np.loadtxt(path)
    if ev.ndim == 1:
        ev = ev.reshape(1, -1)
    if ev.shape[1] != 4:
        raise ValueError(f"Expected 4 columns (t x y p) in {path}, got shape {ev.shape}")
    t = ev[:,0].astype(np.float64)
    x = ev[:,1].astype(np.int32)
    y = ev[:,2].astype(np.int32)
    p = ev[:,3].astype(np.int8)
    return t, x, y, p

# ----------------------------
# 3) Pre-bucket events by frame index (fast)
#    For each event time t, find frame_bin j such that
#    frame_times[j] < t <= frame_times[j+1]
# ----------------------------
def compute_event_frame_bins(t, frame_times):
    # searchsorted returns index in [0..len] for insertion to keep order
    # We want right bin edge, so side='right'
    # For t in (frame_times[j], frame_times[j+1]] -> idx = j+1
    idx = np.searchsorted(frame_times, t, side='right')  # in [0..N]
    # Valid interval bins are 1..N-1; subtract 1 to get j in [0..N-2] where t in (times[j], times[j+1]]
    frame_bin = idx - 1
    return frame_bin  # may be -1 for t <= times[0], or N-1 for t > times[-1]

# ----------------------------
# 4) Build stacks [T,H,W,2] for counts and last-times
#    Using precomputed frame_bin per event:
#    For a starting index i, an event belongs to local interval k = frame_bin - i,
#    keep if 0 <= k < T.
# ----------------------------
def build_event_stacks_for_start(i, T, H, W, frame_times, t, x, y, p, frame_bin):
    count_stack = np.zeros((T, H, W, 2), dtype=np.uint16)
    time_stack  = np.zeros((T, H, W, 2), dtype=np.float32)
    # Select events with local interval k in [0, T-1]
    k = frame_bin - i
    valid = (k >= 0) & (k < T)
    if not np.any(valid):
        image_times = frame_times[i:i+T+1].astype(np.float64)
        return count_stack, time_stack, image_times

    kv = k[valid]
    xv = x[valid]; yv = y[valid]; tv = t[valid]; pv = p[valid]

    # Clamp coordinates
    inb = (xv >= 0) & (xv < W) & (yv >= 0) & (yv < H)
    if not np.any(inb):
        image_times = frame_times[i:i+T+1].astype(np.float64)
        return count_stack, time_stack, image_times

    kv = kv[inb]; xv = xv[inb]; yv = yv[inb]; tv = tv[inb]; pv = pv[inb]

    # Polarity channels: 0 = pos, 1 = neg
    pos = pv > 0
    neg = ~pos

    # Counts
    if np.any(pos):
        # add.at on 3D with index: (k, y, x)
        np.add.at(count_stack, (kv[pos], yv[pos], xv[pos], np.zeros_like(kv[pos])), 1)
    if np.any(neg):
        np.add.at(count_stack, (kv[neg], yv[neg], xv[neg], np.ones_like(kv[neg])), 1)

    # Last times (max per pixel in interval/polarity)
    # We'll rasterize to flat indices for speed, then do maximum.at
    # flat index for HxW: idx = y*W + x
    if np.any(pos):
        flat_pos = yv[pos] * W + xv[pos]
        chan = 0
        # For each interval k, we need a separate buffer to do maximum.at
        for kk in np.unique(kv[pos]):
            m = (kv[pos] == kk)
            if not np.any(m): continue
            tmp = np.zeros(H*W, dtype=np.float32)
            np.maximum.at(tmp, flat_pos[m], tv[pos][m].astype(np.float32))
            time_stack[kk, :, :, chan] = tmp.reshape(H, W)
    if np.any(neg):
        flat_neg = yv[neg] * W + xv[neg]
        chan = 1
        for kk in np.unique(kv[neg]):
            m = (kv[neg] == kk)
            if not np.any(m): continue
            tmp = np.zeros(H*W, dtype=np.float32)
            np.maximum.at(tmp, flat_neg[m], tv[neg][m].astype(np.float32))
            time_stack[kk, :, :, chan] = tmp.reshape(H, W)

    image_times = frame_times[i:i+T+1].astype(np.float64)
    return count_stack, time_stack, image_times

# ----------------------------
# 5) Writer
# ----------------------------
def write_tfrecord_for_bag(root, bag, split, cam, T, frame_times, H, W, t, x, y, p):
    # Precompute frame_bin once (global to all starts)
    frame_bin = compute_event_frame_bins(t, frame_times)  # length M
    # Valid starts require i+T <= N-1  -> i in [0 .. N-1-T]
    N = len(frame_times)
    max_start = N - (T + 0)
    if max_start <= 0:
        raise RuntimeError(f"Not enough frames for T={T}. Need at least {T+1} frames.")

    bag_dir = os.path.join(root, bag)
    os.makedirs(bag_dir, exist_ok=True)

    # TFRecord filename must match the loader
    tfrec_path = os.path.join(bag_dir, f"{cam}_event_images.tfrecord")
    with tf.io.TFRecordWriter(tfrec_path) as w:
        shape_u16 = np.array([T, H, W, 2], dtype=np.uint16)
        for i in range(max_start):
            count_stack, time_stack, image_ts = build_event_stacks_for_start(
                i, T, H, W, frame_times, t, x, y, p, frame_bin
            )
            ex = tf.train.Example(features=tf.train.Features(feature={
                'image_iter':            _int64_feature(i),
                'shape':                 _bytes_feature(shape_u16.tobytes()),
                'event_count_images':    _bytes_feature(count_stack.tobytes()),
                'event_time_images':     _bytes_feature(time_stack.astype(np.float32).tobytes()),
                'image_times':           _bytes_feature(image_ts.astype(np.float64).tobytes()),
                'prefix':                _bytes_feature(bag.encode('utf-8')),
                'cam':                   _bytes_feature(cam.encode('utf-8')),
            }))
            w.write(ex.SerializeToString())

    # n_images.txt expects "N_left N_right".
    # Your loader subtracts _MAX_SKIP_FRAMES (=6) internally when counting usable samples.
    # Write left = max_start + 6 so that (left - 6) == max_start.
    LEFT_PAD = 6
    with open(os.path.join(bag_dir, 'n_images.txt'), 'w') as f:
        f.write(f"{max_start + LEFT_PAD} 0")

    # split list file
    split_file = os.path.join(root, f"{split}_bags.txt")
    with open(split_file, 'w') as f:
        f.write(bag + "\n")

    print(f"[OK] Wrote TFRecord: {tfrec_path}")
    print(f"[OK] n_images.txt with counts: left={max_start + LEFT_PAD} right=0")
    print(f"[OK] {split}_bags.txt lists: {bag}")
    print(f"[OK] Samples (starts) available for T={T}: {max_start}")

# ----------------------------
# 6) Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Build EV-Flow-style TFRecords from v2e TXT and original video.")
    ap.add_argument("--video",default="../gmflow/photo/4369_event/video_orig.avi", help="Path to the original video used for v2e.")
    ap.add_argument("--events_txt", default="../gmflow/photo/4369_event/4369_events.txt", help="Path to v2e events text (t x y p).")
    ap.add_argument("--root", default="./data/custom_data/", help="Dataset root directory to write into.")
    ap.add_argument("--bag", default="bag", help="Bag/folder name under root.")
    ap.add_argument("--split", default="test", choices=["train","val","test"], help="Which split file to write.")
    ap.add_argument("--cam", default="left", help="Camera name used in filenames, default 'left'.")
    ap.add_argument("--T", type=int, default=4, help="#intervals per sample. For test with skip_frames=True, use 4. For skip_frames=False, use 1.")
    ap.add_argument("--target_hw", type=str, default="", help="Optional resize to HxW, e.g. 260x346. Leave empty to keep original.")
    args = ap.parse_args()

    # 1) Extract frames -> PNGs & times
    bag_dir = os.path.join(args.root, args.bag)
    frame_times, H, W = extract_frames_to_png(args.video, bag_dir, cam=args.cam)

    # Optional resize to target HxW (resave PNGs)
    if args.target_hw:
        try:
            tH, tW = map(int, args.target_hw.lower().split('x'))
        except Exception:
            raise ValueError("target_hw must be like '260x346'")
        if (tH, tW) != (H, W):
            print(f"[info] Resizing PNGs from {(H,W)} -> {(tH,tW)} ...")
            for i in range(len(frame_times)):
                p = os.path.join(bag_dir, f"{args.cam}_image{i:05d}.png")
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (tW, tH), interpolation=cv2.INTER_AREA)
                cv2.imwrite(p, img)
            H, W = tH, tW

    # 2) Load events
    t, x, y, p = load_events_txt(args.events_txt)

    # 3) Write TFRecord + indexes
    write_tfrecord_for_bag(
        root=args.root,
        bag=args.bag,
        split=args.split,
        cam=args.cam,
        T=args.T,
        frame_times=frame_times,
        H=H, W=W,
        t=t, x=x, y=y, p=p
    )

    print("\n[Next steps]")
    print(f" - In your loader, call get_loader(root='{args.root}', split='{args.split}', ...)")
    print(f" - For test with T=4, call get_loader(..., split='test', skip_frames=True)")
    print(f" - For test with T=1, rebuild with --T 1 and call get_loader(..., skip_frames=False)")

if __name__ == "__main__":
    main()
