# import onnx
#
# # Đường dẫn đến tệp mô hình ONNX
# onnx_model_path = "resnet50.onnx"
#
# # Nạp mô hình ONNX
# onnx_model = onnx.load(onnx_model_path)
#
# # Kiểm tra tính hợp lệ của mô hình
# onnx.checker.check_model(onnx_model)

import tensorrt as trt

def is_serialized_engine(engine_path):
    with open(engine_path, "rb") as engine_file:
        try:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            engine_data = engine_file.read()
            _ = runtime.deserialize_cuda_engine(engine_data)
            return True
        except Exception as e:
            print(f"Error: {str(e)}")
            return False

engine_path = 'models/cnnv6.engine'  # Thay đổi thành đường dẫn của tệp .engine của bạn
is_valid = is_serialized_engine(engine_path)

if is_valid:
    print(f"The engine file '{engine_path}' is a valid TensorRT engine.")
else:
    print(f"The engine file '{engine_path}' is not a valid TensorRT engine.")


