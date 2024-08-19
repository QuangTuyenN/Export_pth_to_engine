import torch
import torch.nn as nn
import torch.nn.functional as fu

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

device = torch.device('cuda')
char_recog_path = 'cnnv6.pth'
model = Net().to(device)
model.load_state_dict(torch.load(char_recog_path))
model.eval()

dummy_input = torch.randn([8, 1, 224, 224], dtype=torch.float32)  # Example input shape
dummy_input = dummy_input.to(device)
torch.onnx.export(model, dummy_input, "cnnv6.onnx", verbose=True)

