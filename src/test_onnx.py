#!/usr/bin/env python
import os
import time
import numpy as np
import onnxruntime as ort
import cv2

# --- your local modules ---
from config import *
from data_loader import get_loader
from eval_utils import *
from vis_utils import *   # uses flow_viz_np, etc.

import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # keep TF1 for the input pipeline

# ====== CONFIG: point to your ONNX model ======
# Default can be overridden via CLI configs() if you add an arg there.
DEFAULT_ONNX_PATH = "../evflownet_nhwc.onnx"  # expects NHWC [1,H,W,4]

# The 4 ONNX outputs (multi-scale). We'll pick the last (finest).
ONNX_OUTPUTS = [
    "vs/vs/decoder/conv2d_1/BiasAdd:0",
    "vs/vs/decoder/conv2d_3/BiasAdd:0",
    "vs/vs/decoder/conv2d_5/BiasAdd:0",
    "vs/vs/decoder/conv2d_7/BiasAdd:0",
]

def drawImageTitle(img, title):
    cv2.putText(img, title, (60, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                thickness=2, bottomLeftOrigin=False)
    return img

def test_onnx(sess, args, event_image_loader, prev_image_loader, next_image_loader, timestamp_loader):
    """
    Runs evaluation using ONNX Runtime for the EV-FlowNet core.
    Keeps TF1 session ONLY for data loading via your existing queues.
    """

    # ---- ONNX Runtime session ----
    onnx_path = getattr(args, "onnx_path", None) or DEFAULT_ONNX_PATH
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    ort_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_inp_name = ort_sess.get_inputs()[0].name
    onnx_out_names = [o.name for o in ort_sess.get_outputs()]
    # If your ONNX saved different names, fall back to model outputs order:
    if set(ONNX_OUTPUTS).issubset(set(onnx_out_names)):
        out_names = ONNX_OUTPUTS
    else:
        out_names = onnx_out_names  # use whatever is present

    print("ONNX input:", onnx_inp_name, "shape:", ort_sess.get_inputs()[0].shape)
    print("ONNX outputs:", out_names)

    # ---- TF queue runners for data loader ----
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

    # ---- viz helpers ----
    color_wheel_rgb = draw_color_wheel_np(args.image_width, args.image_height)

    # ---- metrics aggregators ----
    max_flow_sum = 0.0
    min_flow_sum = 0.0
    iters = 0

    if args.test_plot:
        cv2.namedWindow('EV-FlowNet Results', cv2.WINDOW_NORMAL)

    if args.gt_path:
        print(f"Loading ground truth {args.gt_path}")
        gt = np.load(args.gt_path)
        gt_timestamps = gt['timestamps']
        U_gt_all = gt['x_flow_dist']
        V_gt_all = gt['y_flow_dist']
        print("Ground truth loaded")
        AEE_sum = 0.0
        percent_AEE_sum = 0.0
        AEE_list = []

    if args.save_test_output:
        output_flow_list = []
        gt_flow_list = []
        event_image_list = []

    try:
        while not coord.should_stop():
            start_time = time.time()
            try:
                # Pull a batch (batch size should be 1 per your get_loader call)
                prev_image, next_image, event_image, image_timestamps = sess.run([
                    prev_image_loader,
                    next_image_loader,
                    event_image_loader,
                    timestamp_loader
                ])
            except tf.errors.OutOfRangeError:
                break

            # event_image: expected shape [1,H,W,4], float32; values likely 0..255 (match training)
            # If your loader produced a different range, adjust here.
            onnx_input = event_image  # NHWC already

            # ---- ONNX inference ----
            outs = ort_sess.run(out_names, {onnx_inp_name: onnx_input})
            # Pick finest head = the one with largest HxW; typically the last
            finest_idx = int(np.argmax([np.prod(o.shape[2:4]) if (o.ndim == 4) else 0 for o in outs]))
            flow_nchw = outs[finest_idx]                 # [1,2,H,W]
            pred_flow = np.transpose(flow_nchw[0], (1, 2, 0))  # -> [H,W,2] (u,v)

            network_duration = time.time() - start_time

            max_flow_sum += np.max(pred_flow)
            min_flow_sum += np.min(pred_flow)

            # Build event count image like your TF version (sum the first 2 channels)
            # event_image is NHWC [1,H,W,4]
            event_count_image = np.sum(event_image[..., :2], axis=-1)  # [1,H,W]
            # Normalize for display:
            ec = event_count_image[0]
            ec_max = np.max(ec) if np.max(ec) > 0 else 1.0
            event_count_image_u8 = (ec * 255.0 / ec_max).astype(np.uint8)

            if args.save_test_output:
                output_flow_list.append(pred_flow)
                event_image_list.append(event_count_image_u8)

            # ---- GT evaluation (optional) ----
            if args.gt_path:
                U_gt, V_gt = estimate_corresponding_gt_flow(U_gt_all, V_gt_all,
                                                            gt_timestamps,
                                                            image_timestamps[0][0],
                                                            image_timestamps[0][1])
                gt_flow = np.stack((U_gt, V_gt), axis=2)  # [H_gt,W_gt,2]

                if args.save_test_output:
                    gt_flow_list.append(gt_flow)

                # center-crop GT to prediction size
                ph, pw = pred_flow.shape[:2]
                gh, gw = gt_flow.shape[:2]
                xoff = (gw - pw) // 2
                yoff = (gh - ph) // 2
                gt_flow_c = gt_flow[yoff:yoff+ph, xoff:xoff+pw, :]

                # Dense flow error
                AEE, percent_AEE, n_points = flow_error_dense(gt_flow_c,
                                                              pred_flow,
                                                              event_count_image_u8,
                                                              'outdoor' in args.test_sequence)
                AEE_list.append(AEE)
                AEE_sum += AEE
                percent_AEE_sum += percent_AEE

            iters += 1
            if iters % 100 == 0:
                print('-------------------------------------------------------')
                print('Iter: {}, time: {:f}, run time: {:.3f}s\n'
                      'Mean max flow: {:.2f}, mean min flow: {:.2f}'
                      .format(iters, image_timestamps[0][0], network_duration,
                              max_flow_sum / iters, min_flow_sum / iters))
                if args.gt_path:
                    print('Mean AEE: {:.2f}, mean %AEE: {:.2f}, # pts: {:.2f}'
                          .format(AEE_sum / iters,
                                  percent_AEE_sum / iters,
                                  n_points))

            # ---- Visualization ----
            if args.test_plot:
                pred_flow_rgb = flow_viz_np(pred_flow[..., 0], pred_flow[..., 1])
                pred_flow_rgb = drawImageTitle(pred_flow_rgb, 'Predicted Flow')

                # Timestamp image (channels 2..3); max over channels like original
                ts_img = np.squeeze(np.amax(event_image[0, ..., 2:], axis=-1))
                ts_max = np.max(ts_img) if np.max(ts_img) > 0 else 1.0
                ts_u8 = (ts_img * 255.0 / ts_max).astype(np.uint8)
                ts_u8 = np.tile(ts_u8[..., np.newaxis], [1, 1, 3])

                # Count image (already computed)
                ec3 = np.tile(event_count_image_u8[..., np.newaxis], [1, 1, 3])

                ts_u8 = drawImageTitle(ts_u8, 'Timestamp Image')
                ec3 = drawImageTitle(ec3, 'Count Image')

                prev_g = np.squeeze(prev_image[0])  # [H,W]
                prev_g = np.tile(prev_g[..., np.newaxis], [1, 1, 3]).astype(np.uint8)
                prev_g = drawImageTitle(prev_g, 'Grayscale Image')

                gt_flow_rgb = np.zeros_like(pred_flow_rgb, dtype=np.uint8)
                errors = np.zeros_like(pred_flow_rgb, dtype=np.uint8)
                gt_flow_rgb = drawImageTitle(gt_flow_rgb, 'GT Flow - No GT')
                errors = drawImageTitle(errors, 'Flow Error - No GT')

                if args.gt_path:
                    # compute error map for viz using cropped GT
                    err = np.linalg.norm(gt_flow_c - pred_flow, axis=-1)
                    em = np.max(err) if np.max(err) > 0 else 1.0
                    err_u8 = (err * 255.0 / em).astype(np.uint8)
                    err_u8 = np.tile(err_u8[..., np.newaxis], [1, 1, 3])
                    err_u8[event_count_image_u8 == 0] = 0
                    if 'outdoor' in args.test_sequence:
                        err_u8[190:, :] = 0  # keep legacy behavior
                    gt_flow_rgb = flow_viz_np(gt_flow_c[..., 0], gt_flow_c[..., 1])
                    gt_flow_rgb = drawImageTitle(gt_flow_rgb, 'GT Flow')
                    errors = drawImageTitle(err_u8, 'Flow Error')

                top_cat = np.concatenate([ec3, prev_g, pred_flow_rgb], axis=1)
                bottom_cat = np.concatenate([ts_u8, errors, gt_flow_rgb], axis=1)
                cat = np.concatenate([top_cat, bottom_cat], axis=0).astype(np.uint8)
                cv2.imshow('EV-FlowNet Results', cat)
                cv2.waitKey(1)

    finally:
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=2)

    print('Testing done.')
    if args.gt_path:
        print('mean AEE {:.2f}, mean %AEE {:.2f}'.format(AEE_sum / iters, percent_AEE_sum / iters))
    if args.save_test_output:
        if args.gt_path:
            print('Saving data to {}_output_gt.npz'.format(args.test_sequence))
            np.savez('{}_output_gt.npz'.format(args.test_sequence),
                     output_flows=np.stack(output_flow_list, axis=0),
                     gt_flows=np.stack(gt_flow_list, axis=0),
                     event_images=np.stack(event_image_list, axis=0))
        else:
            print('Saving data to {}_output.npz'.format(args.test_sequence))
            np.savez('{}_output.npz'.format(args.test_sequence),
                     output_flows=np.stack(output_flow_list, axis=0),
                     event_images=np.stack(event_image_list, axis=0))

def main():
    args = configs()

    # NOTE: we no longer restore TF checkpoints; we use an ONNX file instead.
    # Add an optional arg to your configs() for --onnx_path; else we use default.
    if not hasattr(args, "onnx_path"):
        args.onnx_path = DEFAULT_ONNX_PATH

    # Data loader as before (batch_size=1 recommended)
    event_image_loader, prev_image_loader, next_image_loader, timestamp_loader, n_ima = get_loader(
        args.data_path,
        1,
        args.image_width,
        args.image_height,
        split='test',
        shuffle=False,
        sequence=args.test_sequence,
        skip_frames=args.test_skip_frames,
        time_only=args.time_only,
        count_only=args.count_only)

    print("Read {} images".format(n_ima))

    # Minimal TF session only for the input pipeline
    sess = tf.compat.v1.Session()
    try:
        test_onnx(sess, args,
                  event_image_loader, prev_image_loader, next_image_loader, timestamp_loader)
    finally:
        sess.close()

if __name__ == "__main__":
    main()
