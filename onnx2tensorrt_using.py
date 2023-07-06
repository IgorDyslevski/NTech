import onnx
import onnx_tensorrt.backend as backend
import numpy as np
from datetime import datetime

model = onnx.load("models/yolov8n.onnx")
engine = backend.prepare(model, device='CUDA:0')
input_data = np.random.random(size=(1, 3, 640, 640)).astype(np.float32)
cnt = 1
start = datetime.now()
for i in range(cnt):
    output_data = engine.run(input_data)[0]
    print(f'{i + 1} complete')
finish = datetime.now()
print(output_data)
print(output_data.shape)
print((finish - start) / cnt)

