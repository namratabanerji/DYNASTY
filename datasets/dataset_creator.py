# dataset_creator.py
import h5py
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            # Ensure proper shape and detach
            return data * torch.tensor(self.std, device=data.device).view(1, 1, 1, -1) + \
                   torch.tensor(self.mean, device=data.device).view(1, 1, 1, -1)
        else:
            return data * self.std + self.mean
'''
class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
'''        
class ScaledWrapper(Dataset):
        def __init__(self, base_ds, scaler):
            self.base_ds = base_ds
            self.scaler = scaler
        def __len__(self):
            return len(self.base_ds)
        def __getitem__(self, idx):
            X, A, Y = self.base_ds[idx]
            X_scaled = self.scaler.transform(X.numpy()[:, :, 0])
            X_scaled = torch.tensor(X_scaled[..., np.newaxis], dtype=torch.float32)
            return X_scaled, A, Y
            
class DynamicGraphDataset(Dataset):
    def __init__(self, X, Y, A):
        self.X = X
        self.Y = Y
        self.A = A

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.A[idx], self.Y[idx]

def load_brain_dataset(npy_path, batch_size=32, seq_len=12, pred_len=8):
    abide_data = np.load(npy_path, allow_pickle=True).item()

    window_size = 20
    stride = 1
    threshold = 0.8
    all_node_attrs = []
    all_graphs = []

    for sub in range(len(abide_data['timeseires'])):
        if abide_data['label'][sub] != 0:
            continue

        ts = abide_data['timeseires'][sub]
        N, T = ts.shape
        node_attrs = []
        graphs = []

        for t in range(window_size - 1, T, stride):
            window = ts[:, t - window_size + 1 : t + 1]
            corr = np.corrcoef(window)
            adj = (np.abs(corr) >= threshold).astype(int)
            np.fill_diagonal(adj, 0)
            graphs.append(adj)
            node_attrs.append(ts[:, t])

        all_node_attrs.append(np.stack(node_attrs))
        all_graphs.append(np.stack(graphs))

    chunk_size = seq_len + pred_len
    final_node_attrs, final_graphs = [], []

    for attrs, graphs in zip(all_node_attrs, all_graphs):
        attrs = attrs[:80]
        graphs = graphs[:80]
        for i in range(0, 80, chunk_size):
            attr_chunk = attrs[i:i+chunk_size]
            graph_chunk = graphs[i:i+chunk_size]
            if attr_chunk.shape[0] == chunk_size:
                final_node_attrs.append(attr_chunk[:, :, np.newaxis])
                final_graphs.append(graph_chunk)

    final_node_attrs = np.array(final_node_attrs)
    final_graphs = np.array(final_graphs)

    input_len = seq_len
    output_len = pred_len

    X_seq, Y_seq, A_seq = [], [], []
    for t in range(len(final_node_attrs)):
        X_seq.append(final_node_attrs[t, :input_len])
        Y_seq.append(final_node_attrs[t, input_len:input_len+output_len])
        A_seq.append(final_graphs[t, :input_len])

    X = np.stack(X_seq)
    Y = np.stack(Y_seq)
    A = np.stack(A_seq)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    A_tensor = torch.tensor(A, dtype=torch.float32)

    dataset = DynamicGraphDataset(X_tensor, Y_tensor, A_tensor)

    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.2 * n_total)
    n_test = n_total - n_train - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_values = torch.cat([ds[0] for ds in train_ds], dim=0).numpy()[:, :, 0]
    scaler = StandardScaler(train_values.mean(), train_values.std())

    

    train_loader = DataLoader(ScaledWrapper(train_ds, scaler), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ScaledWrapper(val_ds, scaler), batch_size=batch_size)
    test_loader = DataLoader(ScaledWrapper(test_ds, scaler), batch_size=batch_size)

    return train_loader, val_loader, test_loader, scaler


'''
def load_brain_dataset(
    npy_path: str,
    window_size: int = 20,
    stride: int = 1,
    threshold: float = 0.8,
    input_len: int = 12,
    pred_len: int = 8,
    chunk_size: int = 20,
    batch_size: int = 32,
    val_size: int = 395,
    device: str = "cpu",
):
    abide_data = np.load(npy_path, allow_pickle=True).item()

    all_node_attrs, all_graphs = [], []
    for sub in range(len(abide_data['timeseires'])):
        if abide_data['label'][sub] != 0:
            continue

        ts = abide_data['timeseires'][sub]  # shape: [N, T]
        N, T = ts.shape
        node_attrs, graphs = [], []

        for t in range(window_size - 1, T, stride):
            window = ts[:, t - window_size + 1 : t + 1]
            corr = np.corrcoef(window)
            adj = (np.abs(corr) >= threshold).astype(int)
            np.fill_diagonal(adj, 0)
            graphs.append(adj)
            node_feat = ts[:, t]
            node_attrs.append(node_feat)

        all_node_attrs.append(np.stack(node_attrs))
        all_graphs.append(np.stack(graphs))

    final_node_attrs, final_graphs = [], []
    for attrs, graphs in zip(all_node_attrs, all_graphs):
        attrs = attrs[:80]
        graphs = graphs[:80]
        for i in range(0, 80, chunk_size):
            attr_chunk = attrs[i:i+chunk_size][:, :, np.newaxis]
            graph_chunk = graphs[i:i+chunk_size]
            final_node_attrs.append(attr_chunk)
            final_graphs.append(graph_chunk)

    X_seq, Y_seq, A_seq = [], [], []
    final_node_attrs = np.array(final_node_attrs)
    final_graphs = np.array(final_graphs)
    for t in range(len(final_node_attrs)):
        X_seq.append(final_node_attrs[t, :input_len])
        Y_seq.append(final_node_attrs[t, input_len:input_len+pred_len])
        A_seq.append(final_graphs[t, :input_len])

    X = np.stack(X_seq)
    Y = np.stack(Y_seq)
    A = np.stack(A_seq)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    A_tensor = torch.tensor(A, dtype=torch.float32)

    split_idx = round(len(X_tensor) * 0.7)
    X_train, X_test = X_tensor[:split_idx], X_tensor[split_idx:]
    Y_train, Y_test = Y_tensor[:split_idx], Y_tensor[split_idx:]
    A_train, A_test = A_tensor[:split_idx], A_tensor[split_idx:]

    X_val, X_test = X_test[:val_size], X_test[val_size:]
    Y_val, Y_test = Y_test[:val_size], Y_test[val_size:]
    A_val, A_test = A_test[:val_size], A_test[val_size:]

    scaler = StandardScaler(X_train.mean(), X_train.std())

    train_loader = DataLoader(DynamicGraphDataset(scaler.transform(X_train), Y_train, A_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(DynamicGraphDataset(scaler.transform(X_val), Y_val, A_val), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(DynamicGraphDataset(scaler.transform(X_test), Y_test, A_test), batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader, scaler
'''

#METR-LA dataset creation
def generate_dynamic_adjacency(adj_static, T, drop_rate=0.1):
    N = adj_static.shape[0]
    adj_sequence = []
    for _ in range(T):
        adj = adj_static.copy()
        mask = np.random.rand(N, N) < drop_rate
        adj[mask] = 0
        np.fill_diagonal(adj, 0)
        adj_sequence.append(adj)
    return np.array(adj_sequence)

class TrafficDataset(Dataset):
    def __init__(self, speed_data, adj_mx, seq_len=12, pred_len=12, scaler=None):
        self.speed = speed_data
        self.adj = adj_mx
        self.L = seq_len
        self.H = pred_len
        self.T = self.speed.shape[0]
        self.N = self.speed.shape[1]
        self.n_samples = self.T - self.L - self.H + 1
        self.scaler = scaler

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.speed[idx: idx + self.L]               # (L, N)
        y = self.speed[idx + self.L: idx + self.L + self.H]  # (H, N)
        A_seq = generate_dynamic_adjacency(self.adj, T=self.L)
        if self.scaler:
            x = self.scaler.transform(x)
        X = torch.tensor(x[..., np.newaxis], dtype=torch.float32)
        Y = torch.tensor(y[..., np.newaxis], dtype=torch.float32)
        A = torch.tensor(A_seq, dtype=torch.float32)
        return X, A, Y


def load_metr_dataset(speed_path, adj_path, seq_len=12, pred_len=12, batch_size=32):
    with h5py.File(speed_path, 'r') as f:
        speed_data = f['df']['block0_values'][:]  # shape: [T_total, N]

    with open(adj_path, "rb") as f:
        _, _, adj_mx = pickle.load(f, encoding='latin1')

    dataset = TrafficDataset(speed_data, adj_mx, seq_len=seq_len, pred_len=pred_len)

    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.2 * n_total)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    # Extract training speeds and compute scaler
    train_speeds = torch.cat([x[0] for x in train_ds], dim=0).numpy()[:, :, 0]  # shape: [total_L, N]
    scaler = StandardScaler(train_speeds.mean(), train_speeds.std())
    '''
    class ScaledWrapper(Dataset):
        def __init__(self, base_ds, scaler):
            self.base_ds = base_ds
            self.scaler = scaler
        def __len__(self):
            return len(self.base_ds)
        def __getitem__(self, idx):
            X, A, Y = self.base_ds[idx]
            X_scaled = self.scaler.transform(X.numpy()[:, :, 0])
            X_scaled = torch.tensor(X_scaled[..., np.newaxis], dtype=torch.float32)
            return X_scaled, A, Y
    '''
    train_loader = DataLoader(ScaledWrapper(train_ds, scaler), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ScaledWrapper(val_ds, scaler), batch_size=batch_size)
    test_loader = DataLoader(ScaledWrapper(test_ds, scaler), batch_size=batch_size)

    return train_loader, val_loader, test_loader, scaler


def load_pems_dataset(speed_path, adj_path, seq_len=12, pred_len=12, batch_size=32):
    with open(speed_path, 'rb') as f:
        speed_data = np.array(pickle.load(f))
    
    with open(adj_path, 'rb') as f:
        adj_mx = np.array(pickle.load(f))

    dataset = TrafficDataset(speed_data, adj_mx, seq_len=seq_len, pred_len=pred_len)

    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.2 * n_total)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    # Extract training speeds and compute scaler
    train_speeds = torch.cat([x[0] for x in train_ds], dim=0).numpy()[:, :, 0]  # shape: [total_L, N]
    scaler = StandardScaler(train_speeds.mean(), train_speeds.std())
    '''
    class ScaledWrapper(Dataset):
        def __init__(self, base_ds, scaler):
            self.base_ds = base_ds
            self.scaler = scaler
        def __len__(self):
            return len(self.base_ds)
        def __getitem__(self, idx):
            X, A, Y = self.base_ds[idx]
            X_scaled = self.scaler.transform(X.numpy()[:, :, 0])
            X_scaled = torch.tensor(X_scaled[..., np.newaxis], dtype=torch.float32)
            return X_scaled, A, Y
    '''
    train_loader = DataLoader(ScaledWrapper(train_ds, scaler), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ScaledWrapper(val_ds, scaler), batch_size=batch_size)
    test_loader = DataLoader(ScaledWrapper(test_ds, scaler), batch_size=batch_size)

    return train_loader, val_loader, test_loader, scaler

def load_bitcoin_dataset(otc, seq_len=12, pred_len=8, batch_size=32):
    if otc:
        path="datasets/BITCOIN/soc-sign-bitcoinotc.csv"
    else:
        path="datasets/BITCOIN/soc-sign-bitcoinalpha.csv"
    
    class SlidingWindowDataset(torch.utils.data.Dataset):
        def __init__(self, attribute_snapshots, adj_snapshots, window_size, pred_length, scaler=None):
            self.attr = attribute_snapshots
            self.adj  = adj_snapshots
            self.L    = window_size
            self.H    = pred_length
            self.T    = len(self.attr)
            assert len(self.adj) == self.T
            self.n_samples = self.T - self.L - self.H + 1
            self.scaler = scaler

        def __len__(self):
            return self.n_samples

        def __getitem__(self, idx):
            X_hist = np.stack(self.attr[idx : idx + self.L], axis=0)
            Y      = np.stack(self.attr[idx + self.L : idx + self.L + self.H], axis=0)
            A_hist = np.stack(self.adj[idx : idx + self.L], axis=0)
            if self.scaler:
                X_hist = self.scaler.transform(X_hist)
            return (
                torch.tensor(X_hist, dtype=torch.float32),
                torch.tensor(A_hist, dtype=torch.float32),
                torch.tensor(Y, dtype=torch.float32),
            )

    # --- Step 1: Read and bin time ---
    df = pd.read_csv(path, header=None, names=["SOURCE", "TARGET", "RATING", "TIME"]).sort_values(by="TIME")[:6000]
    start_time = df["TIME"].min()
    df["time_bin"] = ((df["TIME"] - start_time) // (24 * 60 * 60)).astype(int)
    time_bins = sorted(df["time_bin"].unique())

    # --- Step 2: Node attributes ---
    given_avg = df.groupby(["time_bin", "SOURCE"])["RATING"].mean().reset_index()
    received_avg = df.groupby(["time_bin", "TARGET"])["RATING"].mean().reset_index()
    given_avg.columns = ["time_bin", "node", "avg_given"]
    received_avg.columns = ["time_bin", "node", "avg_received"]
    node_attributes = pd.merge(given_avg, received_avg, on=["time_bin", "node"], how="outer").fillna(0)

    # --- Step 3: Map nodes and build sequences ---
    all_nodes = np.unique(df[["SOURCE", "TARGET"]].values)
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    num_nodes = len(node_to_idx)

    attribute_snapshots = []
    adj_snapshots = []
    for t in time_bins:
        adj_t = np.zeros((num_nodes, num_nodes))
        for src, tgt in df[df["time_bin"] == t][["SOURCE", "TARGET"]].values:
            adj_t[node_to_idx[src], node_to_idx[tgt]] = 1
        adj_snapshots.append(adj_t)

        attr_t = node_attributes[node_attributes["time_bin"] == t]
        attr_vec = np.zeros((num_nodes, 2))
        for _, row in attr_t.iterrows():
            idx = node_to_idx[row["node"]]
            attr_vec[idx] = [row["avg_given"], row["avg_received"]]
        attribute_snapshots.append(attr_vec)
    print("dataset:", attr_vec.shape)
    # --- Step 4: Create dataset ---
    dataset = SlidingWindowDataset(attribute_snapshots, adj_snapshots, seq_len, pred_len, scaler=None)

    # --- Step 5: Train/val/test split ---
    n_train = int(0.7 * len(dataset))
    n_val   = int(0.2 * len(dataset))
    n_test  = len(dataset) - n_train - n_val

    dataset = SlidingWindowDataset(attribute_snapshots, adj_snapshots, seq_len, pred_len)
    
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])
    
    #train_values = torch.cat([ds[0] for ds in train_ds], dim=0).numpy()[:, :, 0]
    # --- Step 6: Fit scaler on training data using all timepoints and nodes ---
    train_values = torch.cat([ds[0] for ds in train_ds], dim=0).numpy()  # shape: [num_samples * L, N, D]
    scaler = StandardScaler(train_values.mean(axis=(0, 1)), train_values.std(axis=(0, 1)))  # shape: [D]
    
    
    class ScaledWrapper2(Dataset):
        def __init__(self, base_ds, scaler):
            self.base_ds = base_ds
            self.scaler = scaler
        def __len__(self):
            return len(self.base_ds)
        def __getitem__(self, idx):
            X, A, Y = self.base_ds[idx]
            X_scaled = self.scaler.transform(X.numpy())
            X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
            return X_scaled, A, Y
    
    train_loader = DataLoader(ScaledWrapper2(train_ds, scaler), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ScaledWrapper2(val_ds, scaler), batch_size=batch_size)
    test_loader = DataLoader(ScaledWrapper2(test_ds, scaler), batch_size=batch_size)

    return train_loader, val_loader, test_loader, scaler

