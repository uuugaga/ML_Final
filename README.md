# Tabular-Playground-Series---Aug-2022

### 1. Specification of dependencies

Using .ipynb

把 `nn_model.pt`, `train.csv`, `test.csv`, 與`sample_submission`，  
放在與`train.ipynb`, `inference.ipynb`同一層資料夾

### 2. Training code

執行 109550181_Fianl_train 即可

#### Hyperparameters :

    epoch = 350 # epoch numbers
    batch_size = 150 # batch size
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#### Model Structure:

```
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(23, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 1)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.leaky_relu(out)
        out = self.fc2(out)
        out = self.leaky_relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out
```

### 3. Evaluation code

Save result to `submission.csv`

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.load('nn_model.pt', map_location=torch.device(device))
    model = model.to(device)
    model.eval()

    test_loader = test_data_loader()

    answer = np.array([])
    for i, (data, _) in enumerate(test_loader):
        data = Variable(data.float()).to(device)
        output = model(data).to(device)
        output = output.data.cpu().numpy().ravel()
        answer = np.append(answer, output)

    submission = pd.read_csv("sample_submission.csv")
    submission["failure"] = answer
    submission.reset_index(drop=True).to_csv("submission.csv", index=False)

### 4. Pre-trained models

已附在專案中，也可從[連結](https://drive.google.com/file/d/1exJ4QAGHXSYWwXOAkm3r379otC86mroz/view?usp=share_link)下載

### 5. Result

Private score ->**0.59226**

### 6. Reproduce step

照理說可以直接 git clone 整個專案，直接跑 Inference，也可以用 colab。

在 colab 上開啟 109550181_Fianl_inference.ipynb，用 git clone 把整個專案載下來，並把路徑移到專案的資料夾，就可以執行了。

以下的 code 為上述方法的參考範例

```
!git clone https://github.com/uuugaga/ML_Final.git
import os
os.chdir('/content/ML_Final')
```
