import numpy as np, onnxruntime as ort

sess = ort.InferenceSession("evflownet_nhwc.onnx", providers=["CPUExecutionProvider"])
inp  = sess.get_inputs()[0].name
print("Model expects:", sess.get_inputs()[0].shape)  # should show [1,256,256,4] or [-1,256,256,4]

x = np.random.rand(1,256,256,4).astype(np.float32)   # NHWC
outs = sess.run([o.name for o in sess.get_outputs()], {inp: x})
print([o.shape for o in outs])                       # last one should be [1,2,256,256] or NHWC variant
