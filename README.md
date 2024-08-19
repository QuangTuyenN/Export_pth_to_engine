Export_TensorRT_Engine

Project này hướng dẫn cách chuyển đổi model từ dạng .pth (pytorch) sang .onnx trung gian sau đó sang dạng .engine (tensorrt) để tối ưu khi chạy trên GPU CUDA

Model sử dụng ở đây là Model cnnv6.pth (class Net) dùng để phân loại số từ 0 đến 9 trong bảng số xe - lisence plate (Lưu ý đây chỉ là model ví dụ không thể áp dụng vào dự án biển số xe)

Quy trình chuyển đổi gổm có 2 bước chuyển từ .pth sang onnx và chuyển từ .onnx sang .engine

--------------------------- CÀI ĐẶT MÔI TRƯỜNG -----------------------------

Giả sử CUDA 11.8, python = 3.8

Tạo môi trường trong Anaconda

pip install --upgrade pip

pip install wheel

pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

pip install --upgrade tensorrt-cu11 (hoặc -cu12 với cuda 12. )

Tải Tensorrt trên https://docs.nvidia.com/deeplearning/tensorrt về máy local

cd đến thư mục python trong thư mục TensorRT đã tải về (Có dạng: TensorRT-8.6.1.6)

Tùy vào python đang sử dụng (ở đây 3.8), chạy cài các file wheel tương ứng:

pip install tensorrt-8.6.1-cp38-none-win_amd64.whl

pip install tensorrt_lean-8.6.1-cp38-none-win_amd64.whl

pip install tensorrt_dispatch-8.6.1-cp38-none-win_amd64.whl

Cài đặt pycuda:

conda install -c conda-forge pycuda

pip install onnx



