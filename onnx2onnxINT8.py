from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic('models/yolov8n.onnx', 'models/qyolov8n.onnx', weight_type=QuantType.QInt8)
