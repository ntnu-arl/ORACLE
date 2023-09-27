import onnx
# Load the ONNX model
model = onnx.load("rnn_onnx_model.onnx")
# model = onnx.load("cnn_onnx_model_no_dropout.onnx")

# Check that the IR is well formed
print('check model:', onnx.checker.check_model(model))

# Print a human readable representation of the graph
print('Model :\n\n{}'.format(onnx.helper.printable_graph(model.graph)))