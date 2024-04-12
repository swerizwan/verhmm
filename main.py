import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from data_loader import get_dataloaders
from faceformer import Faceformer

def trainer(args, train_loader, dev_loader, model, optimizer, criterion, epoch=100):
    """
    Train the Faceformer model.

    Args:
        args (argparse.Namespace): Arguments passed from the command line.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        dev_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        model (torch.nn.Module): The Faceformer model.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion: Loss function.
        epoch (int): Number of epochs.

    Returns:
        str: Path to the best model.
    """
    save_path = os.path.join(args.dataset, args.save_path)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    best_loss = float('inf')
    best_model_path = None

    for e in range(epoch + 1):
        train_loss = train_epoch(e, train_loader, model, optimizer, criterion, args)
        valid_loss = validate_epoch(e, dev_loader, model, criterion, args)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model_path = os.path.join(save_path, f'best_model.pth')
            torch.save(model.state_dict(), best_model_path)

        print(f"Epoch: {e+1}, Train Loss: {train_loss:.7f}, Validation Loss: {valid_loss:.7f}")

    return best_model_path

def train_epoch(epoch, train_loader, model, optimizer, criterion, args):
    """
    Train the model for one epoch.

    Args:
        epoch (int): Current epoch.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        model (torch.nn.Module): The Faceformer model.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion: Loss function.
        args (argparse.Namespace): Arguments passed from the command line.

    Returns:
        float: Average training loss.
    """
    model.train()
    total_loss = 0.0

    for i, (audio, vertice, template, one_hot, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
        audio, vertice, template, one_hot = audio.to(device=args.device), vertice.to(device=args.device), \
                                            template.to(device=args.device), one_hot.to(device=args.device)
        optimizer.zero_grad()
        loss = model(audio, template, vertice, one_hot, criterion, teacher_forcing=False)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / (i + 1)

@torch.no_grad()
def validate_epoch(epoch, dev_loader, model, criterion, args):
    """
    Validate the model for one epoch.

    Args:
        epoch (int): Current epoch.
        dev_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        model (torch.nn.Module): The Faceformer model.
        criterion: Loss function.
        args (argparse.Namespace): Arguments passed from the command line.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    total_loss = 0.0

    for audio, vertice, template, one_hot, _ in dev_loader:
        audio, vertice, template, one_hot = audio.to(device=args.device), vertice.to(device=args.device), \
                                            template.to(device=args.device), one_hot.to(device=args.device)
        loss = model(audio, template, vertice, one_hot, criterion)
        total_loss += loss.item()

    return total_loss / len(dev_loader)

def test(args, model, test_loader, best_model_path):
    """
    Test the trained model.

    Args:
        args (argparse.Namespace): Arguments passed from the command line.
        model (torch.nn.Module): The Faceformer model.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        best_model_path (str): Path to the best model.
    """
    result_path = os.path.join(args.dataset, args.result_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    model.load_state_dict(torch.load(best_model_path))
    model = model.to(torch.device("cuda"))
    model.eval()

    for audio, vertice, template, one_hot_all, file_name in test_loader:
        audio, vertice, template, one_hot_all = audio.to(device="cuda"), vertice.to(device="cuda"), \
                                                template.to(device="cuda"), one_hot_all.to(device="cuda")
        train_subject = "_".join(file_name[0].split("_")[:-1])
        for iter in range(one_hot_all.shape[-1]):
            condition_subject = args.train_subjects.split(" ")[iter]
            one_hot = one_hot_all[:, iter, :]
            prediction = model.predict(audio, template, one_hot)
            prediction = prediction.squeeze()
            np.save(os.path.join(result_path, f"{file_name[0].split('.')[0]}_condition_{condition_subject}.npy"),
                    prediction.detach().cpu().numpy())

def count_parameters(model):
    """
    Count the number of trainable parameters in the model.

    Args:
        model (torch.nn.Module): The model.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI')
    parser.add_argument("--vertice_dim", type=int, default=5023*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--features", type=int, default=64, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default= "wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=100, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
       " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
       " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
       " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
       " FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
       " FaceTalk_170731_00024_TA")
    args = parser.parse_args()

    # Set device
    assert torch.cuda.is_available(), "CUDA not available"
    args.device = torch.device(args.device)

    # Build model
    model = Faceformer(args)
    print("Model parameters: ", count_parameters(model))

    # Move model to device
    model = model.to(args.device)

    # Load data
    dataset = get_dataloaders(args)

    # Loss function
    criterion = nn.MSELoss()

    # Train the model
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    best_model_path = trainer(args, dataset["train"], dataset["valid"], model, optimizer, criterion, epoch=args.max_epoch)

    # Test the model
    test(args, model, dataset["test"], best_model_path)

if __name__ == "__main__":
    main()
