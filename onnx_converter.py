# cut_and_convert_from_shuffle_batch.py  (TF 1.15 + tf2onnx)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# register contrib kernels used by your graph
import tensorflow.contrib.image as _contrib_image
import tensorflow.contrib.resampler as _contrib_res

import tf2onnx, onnx

FROZEN = "data/log/saver/ev-flownet/frozen_model.pb"
OUT    = "evflownet_nhwc.onnx"

INPUT  = "shuffle_batch:0"  # <- cut here (NHWC)
OUTPUTS = [
    "vs/vs/decoder/conv2d_1/BiasAdd:0",
    "vs/vs/decoder/conv2d_3/BiasAdd:0",
    "vs/vs/decoder/conv2d_5/BiasAdd:0",
    "vs/vs/decoder/conv2d_7/BiasAdd:0",
]

with tf.gfile.GFile(FROZEN, "rb") as f:
    gd = tf.GraphDef(); gd.ParseFromString(f.read())

with tf.Graph().as_default() as g:
    tf.import_graph_def(gd, name="")
    x = g.get_tensor_by_name(INPUT)
    # Lock boundary shape to NHWC [1,256,256,4]
    x.set_shape([1, 256, 256, 4])

    onnx_g = tf2onnx.tfonnx.process_tf_graph(
        g,
        input_names=[x.name],          # treat shuffle_batch:0 as Placeholder
        output_names=OUTPUTS,
        opset=17,                      # good for contrib mappings              # clean up
        # do NOT set inputs_as_nchw — boundary stays NHWC
    )
    model = onnx_g.make_model("evflownet_nhwc_cut")
    onnx.save(model, OUT)
    print("Wrote", OUT)
