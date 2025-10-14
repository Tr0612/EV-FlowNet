# infer_from_pngs_exact.py
import cv2, numpy as np, onnxruntime as ort

MODEL = "evflownet_nhwc.onnx"  # your working model
IMG0  = "./data/mvsec_data/outdoor_day1/left_image00000.png"
IMG1  = "./data/mvsec_data/outdoor_day1/left_image00001.png"
H, W  = 256, 256

def read_gray_float32_0_255(path, size=(W, H)):
    # Read grayscale uint8 (0..255), resize, cast to float32 (still 0..255), just like TF code.
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert im is not None, f"Could not read {path}"
    im = cv2.resize(im, size, interpolation=cv2.INTER_AREA).astype(np.float32)  # 0..255 float32
    return im

# Build 4-channel NHWC [H,W,4].
# If your training stacked differently, swap the order here.
a = read_gray_float32_0_255(IMG0)
b = read_gray_float32_0_255(IMG1)
x4 = np.stack([a, b, a, b], axis=-1)            # [H,W,4]
x  = x4[None, ...]                               # [1,H,W,4]

# Inference
sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
inp  = sess.get_inputs()[0].name
outs = sess.run([o.name for o in sess.get_outputs()], {inp: x})

# Pick finest head (largest HxW); your last output is 256x256
finest = outs[-1]                                # [1,2,256,256] (NCHW)
flow   = np.transpose(finest[0], (1,2,0))        # -> [256,256,2]

# Simple colorization for a quick look
def flow_to_color(flow_uv, clip=None):
    u, v = flow_uv[...,0], flow_uv[...,1]
    rad = np.sqrt(u*u+v*v)
    if clip is None: clip = np.percentile(rad, 99) + 1e-6
    u = np.clip(u/clip, -1, 1); v = np.clip(v/clip, -1, 1)
    ang = (np.arctan2(-v, -u) / np.pi + 1) / 2.0
    mag = np.clip(np.sqrt(u*u+v*v), 0, 1)
    hsv = np.zeros((flow_uv.shape[0], flow_uv.shape[1],3), np.uint8)
    hsv[...,0] = (ang*179).astype(np.uint8)
    hsv[...,1] = 255
    hsv[...,2] = (mag*255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

flow_color = flow_to_color(flow)
cv2.imshow("Flow (color)", flow_color)
cv2.waitKey(0)
