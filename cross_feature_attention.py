import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class CustomerDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class PairFeatureAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(feature_dim, num_heads)
        
    def forward(self, feature_pair):
        """
        feature_pair: [batch_size, 2, feature_dim] (客户A和客户B的同一特征)
        """
        feature_pair = feature_pair.transpose(0, 1)  # [2, batch_size, feature_dim]
        attn_output, attn_weights = self.multihead_attn(
            query=feature_pair,
            key=feature_pair,
            value=feature_pair
        )
        attn_output = attn_output.transpose(0, 1)  # [batch_size, 2, feature_dim]
        
        feature_a_attended = attn_output[:, 0]  # [batch_size, feature_dim]
        feature_b_attended = attn_output[:, 1]  # [batch_size, feature_dim]
        
        return feature_a_attended, feature_b_attended, attn_weights

class CustomerMatchingModel(nn.Module):
    def __init__(self, feature_dim, num_features=4, hidden_dim=64):
        super().__init__()
        self.attention_layers = nn.ModuleList([
            PairFeatureAttention(feature_dim) for _ in range(num_features)
        ])
        
        self.fc1 = nn.Linear(feature_dim * num_features * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, feature_pairs):
        """
        feature_pairs: [batch_size, num_features, 2, feature_dim]
        """
        batch_size = feature_pairs.size(0)
        all_features = []
        
        for i in range(len(self.attention_layers)):
            feature_a, feature_b, _ = self.attention_layers[i](feature_pairs[:, i])
            all_features.append(feature_a)
            all_features.append(feature_b)
        
        concat_features = torch.cat(all_features, dim=1)  # [batch_size, feature_dim*num_features*2]
        
        x = self.fc1(concat_features)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return torch.sigmoid(x)

def generate_sample_data(num_samples=1000, num_features=4, feature_dim=16):
    """生成示例数据用于模型测试"""
    same_person_features = []
    for _ in range(num_samples // 2):
        pairs = []
        for _ in range(num_features):
            feature_a = np.random.randn(feature_dim)
            feature_b = feature_a + np.random.normal(0, 0.1, size=feature_dim)
            pairs.append((feature_a, feature_b))
        same_person_features.append((pairs, 1))
    
    diff_person_features = []
    for _ in range(num_samples // 2):
        pairs = []
        for _ in range(num_features):
            feature_a = np.random.randn(feature_dim)
            feature_b = np.random.randn(feature_dim)
            pairs.append((feature_a, feature_b))
        diff_person_features.append((pairs, 0))
    
    all_data = same_person_features + diff_person_features
    np.random.shuffle(all_data)
    
    features = [data[0] for data in all_data]
    labels = [data[1] for data in all_data]
    
    features_array = np.array(features)  # [num_samples, num_features, 2, feature_dim]
    labels_array = np.array(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        features_array, labels_array, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def train_model(model, train_loader, test_loader, epochs=10, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            features, labels = batch
            labels = labels.float().view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        model.eval()
        test_preds = []
        test_labels = []
        with torch.no_grad():
            for batch in test_loader:
                features, labels = batch
                labels = labels.float().view(-1, 1)
                
                outputs = model(features)
                preds = (outputs > 0.5).float()
                
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
        
        accuracy = np.mean(np.array(test_preds) == np.array(test_labels))
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Test Accuracy: {accuracy:.4f}')
    
    print("\n分类报告:")
    print(classification_report(test_labels, test_preds, target_names=['不同人', '同人']))

if __name__ == "__main__":
    # 生成数据
    X_train, X_test, y_train, y_test = generate_sample_data(
        num_samples=2000, num_features=4, feature_dim=16
    )
    
    # 创建数据集和数据加载器
    train_dataset = CustomerDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_dataset = CustomerDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 初始化模型
    model = CustomerMatchingModel(feature_dim=16, num_features=4, hidden_dim=64)
    
    # 训练模型
    train_model(model, train_loader, test_loader, epochs=10)
    
    # 示例预测
    sample_idx = 0
    sample_features = test_dataset[sample_idx][0].unsqueeze(0)
    model.eval()
    with torch.no_grad():
        prediction = model(sample_features)
        print(f"\n示例预测结果: {prediction.item():.4f} (实际标签: {test_dataset[sample_idx][1]})")
        print("预测为: " + ("同人" if prediction.item() > 0.5 else "不同人"))    