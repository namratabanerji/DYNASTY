# train_eval_utils.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def variation_loss(Y_pred, Y_true):
    pred_diff = Y_pred[:, 1:] - Y_pred[:, :-1]
    true_diff = Y_true[:, 1:] - Y_true[:, :-1]
    return F.mse_loss(pred_diff, true_diff)

def masked_mae(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) * mask
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    mask /= torch.mean(mask)
    loss = (torch.abs(preds - labels) / (labels + 1e-5)) * mask
    return torch.mean(torch.where(torch.isnan(loss), torch.zeros_like(loss), loss))

def masked_mse(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    mask /= torch.mean(mask)
    loss = (preds - labels) ** 2 * mask
    return torch.mean(torch.where(torch.isnan(loss), torch.zeros_like(loss), loss))

def masked_rmse(preds, labels, null_val=0.0):
    return torch.sqrt(masked_mse(preds, labels, null_val))

def plot_predictions(Y_true, Y_pred, epoch, save_dir='plots', max_nodes=5):
    os.makedirs(save_dir, exist_ok=True)
    B, H, N, D = Y_true.shape
    for n in range(min(N, max_nodes)):
        for d in range(min(D, 1)):
            plt.figure(figsize=(6, 3))
            plt.plot(Y_true[0, :, n, d].cpu(), label='True')
            plt.plot(Y_pred[0, :, n, d].cpu(), label='Predicted', linestyle='--')
            plt.title(f"Epoch {epoch} | Node {n}, Feature {d}")
            plt.xlabel("Prediction step")
            plt.ylabel("Value")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"epoch{epoch}_node{n}_feat{d}.png"))
            plt.close()

def apply_mask(X, mask_ratio=0.15):
    mask = (torch.rand_like(X) < mask_ratio).float()
    masked_X = X.clone()
    masked_X[mask.bool()] = 0.0
    return masked_X, mask

def masked_reconstruction_loss(pred, target, mask):
    loss = ((pred - target) ** 2) * mask
    return loss.sum() / (mask.sum() + 1e-6)

def pretrain_masked_model(model, dataloader, device, epochs=10, lr=1e-3, mask_ratio=0.15):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for X_hist, A_hist, _ in tqdm(dataloader, desc=f"Pretrain Epoch {epoch+1}", leave=False):
            X_hist, A_hist = X_hist.to(device), A_hist.to(device)
            X_masked, mask = apply_mask(X_hist, mask_ratio)
            mask = mask.to(device)

            H = model.encode(X_masked, A_hist)
            X_recon = model.output_proj(H)

            loss = masked_reconstruction_loss(X_recon, X_hist, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Pretrain] Epoch {epoch+1}: Loss = {total_loss:.4f}")
    return model

def train_epoch(model, dataloader, optimizer, scaler, device, epoch):
    model.train()
    total_rmse, total_loss, count = 0, 0, 0
    for X_hist, A_hist, Y_true in tqdm(dataloader, desc="Train", leave=False):
        #print("^^^^^^^^^^^^^^^", X_hist.shape)
        X_hist, A_hist, Y_true = X_hist.to(device), A_hist.to(device), Y_true.to(device)
        Y_pred = scaler.inverse_transform(model(X_hist, A_hist, Y_true, epoch))

        fut_len = Y_pred.shape[1]
        if epoch < fut_len:
            fut_len_eff = epoch
        else:
            fut_len_eff = fut_len

        Y_pred_eff = Y_pred[:, :fut_len_eff]
        Y_true_eff = Y_true[:, :fut_len_eff]

        loss = masked_mae(Y_pred_eff, Y_true_eff, 0.0) + 0.1 * variation_loss(Y_pred_eff, Y_true_eff)
        total_rmse += masked_rmse(Y_pred_eff, Y_true_eff, 0.0).item()
        total_loss += loss.item()
        count += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Train RMSE:', total_rmse / count)
    return total_loss / count

def eval_epoch(model, dataloader, scaler, device, epoch, plot = False, val_loader=None):
    model.eval()
    if epoch % 5 == 0 and plot and val_loader is not None:
        with torch.no_grad():
            for X_hist, A_hist, Y_true in val_loader:
                X_hist, A_hist, Y_true = X_hist.to(device), A_hist.to(device), Y_true.to(device)
                Y_pred = scaler.inverse_transform(model(X_hist, A_hist))
                plot_predictions(Y_true, Y_pred, epoch)
                break

    total_rmse, total_mae, total_mape, count = 0, 0, 0, 0
    with torch.no_grad():
        for X_hist, A_hist, Y_true in tqdm(dataloader, desc="Eval", leave=False):
            X_hist, A_hist, Y_true = X_hist.to(device), A_hist.to(device), Y_true.to(device)
            Y_pred = scaler.inverse_transform(model(X_hist, A_hist))

            total_rmse += masked_rmse(Y_pred, Y_true, 0.0).item()
            total_mae += masked_mae(Y_pred, Y_true, 0.0).item()
            total_mape += masked_mape(Y_pred, Y_true, 0.0).item()
            count += 1

    return total_rmse / count, total_mae / count, total_mape / count

def train_model(model, train_loader, val_loader, scaler, device, epochs=100, lr=1e-3, patience=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    best_model = None
    wait = 0
    total_train_time, total_eval_time = 0, 0

    for epoch in range(1, epochs + 1):
        import time
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch)
        total_train_time += time.time() - start

        start = time.time()
        val_rmse, val_mae, val_mape = eval_epoch(model, val_loader, scaler, device, epoch, val_loader)
        total_eval_time += time.time() - start

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.4f}")

        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            best_model = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model)
    return model, total_train_time / epochs, total_eval_time / epochs

def test_model(model, test_loader, scaler, device):
    import time
    start = time.time()
    rmse, mae, mape = eval_epoch(model, test_loader, scaler, device, epoch=0)
    elapsed = time.time() - start
    return elapsed, rmse, mae, mape

