import onnx
m = onnx.load("evflownet_nhwc.onnx")
onnx.checker.check_model(m)
print("OK. Opsets:", [o.version for o in m.opset_import])

print("\nInputs:")
for i in m.graph.input:
    shp = [d.dim_value if d.dim_value>0 else -1 for d in i.type.tensor_type.shape.dim]
    print(" ", i.name, shp)

print("\nOutputs:")
for o in m.graph.output:
    shp = [d.dim_value if d.dim_value>0 else -1 for d in o.type.tensor_type.shape.dim]
    print(" ", o.name, shp)
