import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch
import os
import time
import cv2
import torch.nn as nn
import torch.nn.functional as fu
import torchvision.transforms as transforms

TRT_LOGGER = trt.Logger()  # This logger is required to build an engine


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 112, 5)
        self.fc1 = nn.LazyLinear(out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 31)

    def forward(self, x):
        x = self.pool(fu.relu(self.conv1(x)))
        x = self.pool(fu.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = fu.relu(self.fc1(x))
        x = fu.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=8):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    t1 = time.perf_counter()
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    t2 = time.perf_counter()
    t = t2 -t1
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs], t


def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs

# onnx_model_path = 'resnet50.onnx'
pytorch_model_path = 'cnnv6.pth'

# These two modes are dependent on hardwares
fp16_mode = True
int8_mode = False
trt_engine_path = 'model_fp16_{}_int8_{}.trt'.format(fp16_mode, int8_mode)

transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Grayscale(),
             transforms.Normalize(0.5, 0.5)])
max_batch_size = 8 # The batch size of input mush be smaller the max_batch_size once the engine is built
# Load and preprocess the image
folder_path = './5/'
for file in os.listdir(folder_path):
    image_path = os.path.join(folder_path, file)  # Thay đổi thành đường dẫn của hình ảnh bạn muốn phân loại
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (224, 224))  # Đảm bảo kích thước ảnh phù hợp với mô hình

    # Normalize the image
    image = (image / 255.0).astype(np.float32)
    x_input = image
    # x_input = np.random.rand(max_batch_size, 3, 224, 224).astype(dtype=np.float32)
    # Load the TensorRT engine
    engine_path = 'models/cnnv6.engine'  # Thay đổi thành đường dẫn đến tệp engine của bạn
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(engine_path, 'rb') as engine_file:
        engine_data = engine_file.read()
        engine = runtime.deserialize_cuda_engine(engine_data)

    # Create the context for this engine
    context = engine.create_execution_context()
    # Allocate buffers for input and output
    inputs, outputs, bindings, stream = allocate_buffers(engine) # input, output: host # bindings

    # Do inference engine
    shape_of_output = (max_batch_size, 31)
    # Load data to the buffer
    inputs[0].host = x_input.reshape(-1)
    # inputs[1].host = ... for multiple input

    trt_outputs, t = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream) # numpy data

    output_from_trt_engine = postprocess_the_outputs(trt_outputs[0], shape_of_output)
    # print("output: ", output_from_trt_engine)
    print("class engine: ", np.argmax(output_from_trt_engine))
    print("inference time engine: ", t)

    #  inference pth
    device = torch.device('cuda')
    char_recog_path = 'cnnv6.pth'
    model_char_recog = Net().to(device)
    model_char_recog.load_state_dict(torch.load(char_recog_path))
    model_char_recog.eval()
    with torch.no_grad():
        pil_image = cv2.imread(image_path)
        # transform image and move it to GPU top process
        image = transform(pil_image).to(device)
        image = image.unsqueeze(0)
        t3 = time.perf_counter()
        output = model_char_recog.forward(image)
        t4 = time.perf_counter()
        probabilities = fu.softmax(output, dim=1)
        # get the index of max probabilities
        index = np.argmax(probabilities.cpu().numpy(), axis=1)
        print("class pth: ", index)
        print("time pth: ", t4 - t3)
