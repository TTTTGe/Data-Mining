import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import RandomOverSampler

from tqdm import tqdm
tqdm.pandas()
# Load data
data = pd.read_csv('./mbti_1.csv')

#####################################
############  数据处理   ############
#####################################

# 数据增强，添加一个特征
# 增强前为[type，posts]，增强后为[type，posts，average_length]
def calculate_average_length(data):
    return len(data.split(' ')) / len(data.split('|||'))

data['average_length'] = data['posts'].progress_apply(calculate_average_length)


# 清洗数据，去掉posts中的网址并用单词“link”代替之，去掉奇怪的字符仅保留单词
def preprocess_text(text):
    # Remove links
    temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link', text)
    # Remove non-alphabetic characters
    temp = re.sub("[^a-zA-Z]", " ", temp)
    # Convert to lowercase
    temp = temp.lower()
    # Remove space >1
    temp = re.sub(' +', ' ', temp)
    return temp

# 使用tqdm显示进度条
data['posts'] = data['posts'].progress_apply(preprocess_text)

#####################################
###########  训练和测试   ############
#####################################

# 划分训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 文本向量化
vectorizer = CountVectorizer()
X_train_text = vectorizer.fit_transform(train_data['posts'])
X_test_text = vectorizer.transform(test_data['posts'])

# 获取平均长度特征
X_train_avg_length = train_data['average_length'].values.reshape(-1, 1)
X_test_avg_length = test_data['average_length'].values.reshape(-1, 1)

# 将MBTI类型转换为整数标签
label_map = {'INTJ': 0, 'INTP': 1, 'INFJ': 2, 'INFP': 3, 'ISTJ': 4, 'ISTP': 5, 'ISFJ': 6, 'ISFP': 7,
             'ENTJ': 8, 'ENTP': 9, 'ENFJ': 10, 'ENFP': 11, 'ESTJ': 12, 'ESTP': 13, 'ESFJ': 14, 'ESFP': 15}
train_labels = train_data['type'].map(label_map)
test_labels = test_data['type'].map(label_map)

# 把较少的MBTI种类上采样
ros = RandomOverSampler(random_state=42)
train_data, train_labels = ros.fit_resample(train_data, train_labels)

# 定义数据集类
class MBTIDataset(Dataset):
    def __init__(self, X_text, X_avg_length, y):
        self.X_text = torch.tensor(X_text.toarray(), dtype=torch.float32)
        self.X_avg_length = torch.tensor(X_avg_length, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)
    
    def __len__(self):
        return len(self.X_text)
    
    def __getitem__(self, idx):
        return self.X_text[idx], self.X_avg_length[idx], self.y[idx]

# 创建数据加载器
train_dataset = MBTIDataset(X_train_text, X_train_avg_length, train_labels)
test_dataset = MBTIDataset(X_test_text, X_test_avg_length, test_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

class MBTIModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(MBTIModel, self).__init__()
        self.layer1 = nn.Linear(num_features, 256)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.output_layer = nn.Linear(128, num_classes)
    
    def forward(self, x_text, x_avg_length):
        # Concatenate text features and average length
        x = torch.cat((x_text, x_avg_length), dim=1)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x
    
    # 设定模型参数
num_features = X_train_text.shape[1] + 1  # 文本特征数量 + 平均长度特征
num_classes = len(label_map)

# 实例化模型
model = MBTIModel(num_features, num_classes)

# 移动模型到 GPU，如果可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()  # 设置模型为训练模式
    for epoch in range(num_epochs):
        total_loss = 0
        for X_text, X_avg_length, y in train_loader:
            X_text, X_avg_length, y = X_text.to(device), X_avg_length.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_text, X_avg_length)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}')

train_model(model, train_loader, criterion, optimizer, num_epochs=15)

def evaluate_model(model, test_loader, criterion):
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # 关闭梯度计算，减少内存消耗和加速计算
        for X_text, X_avg_length, y in test_loader:
            X_text, X_avg_length, y = X_text.to(device), X_avg_length.to(device), y.to(device)
            
            outputs = model(X_text, X_avg_length)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == y).sum().item()
            total_predictions += y.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy
# 评估模型
test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')