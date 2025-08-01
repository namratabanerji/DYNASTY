# main.py
import argparse
import torch
import torch
from model import Dynasty
from train_eval_utils import train_model, test_model, pretrain_masked_model
from datasets.dataset_creator import load_brain_dataset, load_metr_dataset, load_pems_dataset, load_bitcoin_dataset

# Recommended parameter values for brain dataset
seq_len = 12 #Do not change this, the dataset was created based on this length
pred_len = 8 #Do not change this, the dataset was created based on this length
batch_size = 32
in_feats = 1 #Do not change this, the dataset is one dimensional for each node
hidden_dim = 56
num_heads = 4
mlp_hidden = 128
num_layers = 4
dropout = 0.1
pretrain_epochs = 15
train_epochs = 50



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["brain", "metr", "pems", "bitcoin-otc", "bitcoin-alpha"], default="brain")
    parser.add_argument("--feat_dim", type=int, default=1)
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--pred_len", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=56)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--mlp_hidden", type=int, default=128)
    parser.add_argument("--pretrain", action='store_true')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.dataset == "brain":
    # Download and save in datasets/BRAIN. Find the dataset here : https://github.com/HennyJie/BrainGB/tree/master/examples/utils/get_abide
        train_loader, val_loader, test_loader, scaler = load_brain_dataset(
            npy_path="./datasets/BRAIN/abide.npy",
            batch_size=args.batch_size,
            seq_len=args.input_len,
            pred_len=args.pred_len
        )
    elif args.dataset == "metr":
     # Download and save in datasets/METR-LA. Find the dataset here: https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset
        train_loader, val_loader, test_loader, scaler = load_metr_dataset(
            speed_path="./datasets/METR-LA/METR-LA.h5",
            adj_path="./datasets/METR-LA/adj_METR-LA.pkl",
            batch_size=args.batch_size,
            seq_len=args.input_len,
            pred_len=args.pred_len
        )
    elif args.dataset == "bitcoin-otc":
    # Download and save in datasets/BITCOIN. Find the dataset here: https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html
        train_loader, val_loader, test_loader, scaler = load_bitcoin_dataset(
            otc = True,
            batch_size=args.batch_size,
            seq_len=args.input_len,
            pred_len=args.pred_len
        )
    elif args.dataset == "bitcoin-alpha":
    # Download and save in datasets/BITCOIN. Find the dataset here: https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html
        train_loader, val_loader, test_loader, scaler = load_bitcoin_dataset(
            otc = False,
            batch_size=args.batch_size,
            seq_len=args.input_len,
            pred_len=args.pred_len
        )
    else:
        train_loader, val_loader, test_loader, scaler = load_pems_dataset(
            speed_path="./datasets/PEMS/pems_speed.pkl",
            adj_path="./datasets/PEMS/pems_adj.pkl",
            batch_size=args.batch_size,
            seq_len=args.input_len,
            pred_len=args.pred_len
        )
    hid = args.hidden_dim
    rem = hid % args.num_heads
    if rem > (args.num_heads - 1) // 2:
        hid = hid + args.num_heads - rem
    else:
        hid = hid - rem
    print(args.feat_dim)
    model = Dynasty(
        in_feats=args.feat_dim,
        hidden_dim=hid,
        num_heads=args.num_heads,
        mlp_hidden=args.mlp_hidden,
        num_layers=args.num_layers,
        hist_len=args.input_len,
        fut_len=args.pred_len
    ).to(args.device)

    if args.pretrain:
        model = pretrain_masked_model(model, train_loader, args.device, epochs=15, lr=args.lr, mask_ratio=0.15)

    model, tr_time, ev_time = train_model(model, train_loader, val_loader, scaler, args.device, epochs=args.epochs, lr=args.lr)
    inf_time, rmse, mae, mape = test_model(model, test_loader, scaler, args.device)

    print("Training Time:", tr_time)
    print("Eval Time:", ev_time)
    print("Inference Time:", inf_time)
    print("Test RMSE:", rmse, "MAE:", mae, "MAPE:", mape)

if __name__ == "__main__":
    main()

"""

import torch
from model import Dynasty
from train_eval_utils import train_model, test_model, pretrain_masked_model
from datasets.dataset_creator import load_brain_dataset, load_metr_dataset

def main():
    # Recommended parameter values for brain dataset
    seq_len = 12 #Do not change this, the dataset was created based on this length
    pred_len = 8 #Do not change this, the dataset was created based on this length
    batch_size = 32
    in_feats = 1 #Do not change this, the dataset is one dimensional for each node
    hidden_dim = 56
    num_heads = 4
    mlp_hidden = 128
    num_layers = 4
    dropout = 0.1
    pretrain_epochs = 15
    train_epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    '''
    train_loader, val_loader, test_loader, scaler = load_brain_dataset(
        npy_path="/users/PAS2041/banerji8/Abide/ABIDE_pcp/abide.npy",
        batch_size=batch_size,
        seq_len=seq_len,
        pred_len=pred_len
    )
    '''
    
    seq_len = 12 
    pred_len = 12 
    batch_size = 32
    in_feats = 1 #Do not change this, the dataset is one dimensional for each node
    hidden_dim = 48
    num_heads = 4
    mlp_hidden = 128
    num_layers = 4
    dropout = 0.1
    pretrain_epochs = 15
    train_epochs = 50
    
    
    train_loader, val_loader, test_loader, scaler = load_metr_dataset(
    speed_path="./datasets/METR-LA/METR-LA.h5",
    adj_path="./datasets/METR-LA/adj_METR-LA.pkl",
    seq_len=12,
    pred_len=12,
    batch_size=32
    )
    
    # Model
    model = Dynasty(
        in_feats=in_feats,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        mlp_hidden=mlp_hidden,
        num_layers=num_layers,
        hist_len=seq_len,
        fut_len=pred_len,
        dropout=dropout
    ).to(device)

    # Optional: Masked Pretraining
    print("Starting masked pretraining...")
    model = pretrain_masked_model(model, train_loader, device, epochs=pretrain_epochs, lr=1e-3, mask_ratio=0.15)

    # Train
    print("Starting training...")
    model, tr_time, ev_time = train_model(model, train_loader, val_loader, scaler, device, epochs=train_epochs, lr=1e-3)

    # Test
    print("Starting testing...")
    inf_time, rmse, mae, mape = test_model(model, test_loader, scaler, device)

    # Report
    print("--- Summary ---")
    print(f"Train Time/Epoch: {tr_time:.2f}s, Eval Time/Epoch: {ev_time:.2f}s")
    print(f"Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}")
    print(f"Inference Time: {inf_time:.2f}s")

if __name__ == '__main__':
    main()
"""
