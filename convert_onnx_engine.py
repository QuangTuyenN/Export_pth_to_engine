import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
network = builder.create_network(flag)
parser = trt.OnnxParser(network, TRT_LOGGER)
config = builder.create_builder_config()

with open("cnnv6.onnx", "rb") as model_file:
    onnx_model = model_file.read()

parser.parse(onnx_model)

engine = builder.build_engine(network, config)

with builder.build_engine(network, config) as engine, open("cnnv6.engine", 'wb') as t:
    t.write(engine.serialize())

runtime = trt.Runtime(TRT_LOGGER)

with open("cnnv6.engine", "rb") as engine_file:
    engine_data = engine_file.read()
    engine = runtime.deserialize_cuda_engine(engine_data)




