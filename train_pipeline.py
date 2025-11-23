import argparse
import json
import os
import time
from typing import Optional, Sequence, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms as T
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, OxfordIIITPet
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import MLP as TorchvisionMLP
import timm


class CachedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, cache_size: int = 0):
        self.base = base_dataset

    def __getitem__(self, idx):
        return self.base[int(idx)]

    def __len__(self):
        return len(self.base)


def get_transforms(model_name: str, dataset_name: str, augmentations: Sequence[str], is_train: bool):
    augmentations = list(augmentations or [])

    resnet_like = {"resnet18","resnet50","resnet14d","resnet26d"}
    if dataset_name == "mnist":
        size = 28
    elif dataset_name.startswith("cifar"):
        size = 32
    else:
        size = 224  # generic large image datasets

    tlist = []
    if is_train:
        if "crop" in augmentations:
            if size <= 32:
                tlist.append(T.RandomCrop(size, padding=4))
            else:
                tlist.append(T.RandomResizedCrop(size, scale=(0.7, 1.0)))
        if "flip" in augmentations:
            tlist.append(T.RandomHorizontalFlip())
        if "rotation" in augmentations:
            tlist.append(T.RandomRotation(10))
        if "translation" in augmentations:
            tlist.append(T.RandomAffine(degrees=0, translate=(0.1,0.1)))
        if "colorjitter" in augmentations:
            tlist.append(T.ColorJitter(0.2,0.2,0.2,0.05))

    tlist.append(T.Resize((size,size)))
    tlist.append(T.ToTensor())

    if dataset_name == "mnist":
        tlist.append(T.Lambda(lambda x: x.expand(3,-1,-1) if hasattr(x, "shape") and x.shape[0]==1 else x))

    if model_name.lower() in resnet_like:
        mean = [0.485,0.456,0.406]; std=[0.229,0.224,0.225]
    elif dataset_name == "mnist":
        mean = [0.1307]; std=[0.3081]
    else:
        mean = [0.4914,0.4822,0.4465]; std=[0.2023,0.1994,0.2010]

    tlist.append(T.Normalize(mean=mean, std=std))
    return T.Compose(tlist)

def get_model(model_name: str, num_classes: int, pretrained=True, device='cuda', dataset_name='cifar-10'):
    name = model_name.lower()
    if name == "mlp":
        if dataset_name == "mnist":
            in_channels, size = 1, 28
        elif dataset_name.startswith("cifar"):
            in_channels, size = 3, 32
        else:
            in_channels, size = 3, 224

        in_features = in_channels * size * size
        hidden_channels = [512, 256]
        mlp = TorchvisionMLP(
            in_channels=in_features,
            hidden_channels=hidden_channels,
            activation_layer=nn.ReLU,
            bias=True,
            dropout=0.0
        )
        model = nn.Sequential(
            nn.Flatten(),
            mlp,
            nn.Linear(hidden_channels[-1], num_classes)
        )
        return model.to(device)

    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


class CombinedOptimizer:
    def __init__(self, optimizers: Iterable[torch.optim.Optimizer]):
        self.optimizers = list(optimizers)

    def step(self, closure=None):
        out = None
        for opt in self.optimizers:
            out = opt.step(closure=closure) if closure is not None else opt.step()
        return out

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            try:
                opt.zero_grad(set_to_none=set_to_none)
            except TypeError:
                opt.zero_grad()

    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            if hasattr(opt, "param_groups"):
                groups.extend(opt.param_groups)
        return groups

    def state_dict(self):
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, sd: dict):
        sds = sd.get("optimizers", [])
        if len(sds) != len(self.optimizers):
            raise ValueError("state_dict optimizers count mismatch")
        for opt, s in zip(self.optimizers, sds):
            opt.load_state_dict(s)

# --------------------------
# Optimizer factory
# --------------------------
def get_optimizer(optim_name: str, model: nn.Module, lr: float, weight_decay: float, momentum: float):
    name = optim_name.lower()

    if name in ("rmsprop", "rms"):
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if name == "muon":
        Muon = None
        try:
            from torch.optim import Muon as _Muon
            Muon = _Muon
        except Exception:
            try:
                from muon import Muon as _Muon2
                Muon = _Muon2
            except Exception:
                Muon = None

        if Muon is None:
            raise RuntimeError("Muon requested but not available in torch.optim or 'muon' package. Install a PyTorch with Muon or the 'muon' package.")

        muon_params = []
        adamw_params = []
        for _, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() == 2:
                muon_params.append(p)
            else:
                adamw_params.append(p)

        if len(muon_params) == 0:
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        muon_opt = Muon(muon_params, lr=lr, weight_decay=weight_decay, momentum=momentum)
        adamw_opt = torch.optim.AdamW(adamw_params, lr=lr, weight_decay=weight_decay)
        return CombinedOptimizer([muon_opt, adamw_opt])

    if name == "sam":
        base_opt_cls = torch.optim.SGD
        rho = 0.05
        class SimpleSAM:
            def __init__(self, params, base_optimizer_cls, rho=0.05, lr=0.01, momentum=0.9, weight_decay=0.0):
                self.params = list(params)
                self.rho = rho
                self.base_optimizer = base_optimizer_cls(self.params, lr=lr, momentum=momentum, weight_decay=weight_decay)
                self.state = {}
            def first_step(self, zero_grad=False):
                grad_norm = torch.norm(torch.stack([p.grad.detach().norm() for p in self.params if p.grad is not None]))
                scale = self.rho / (grad_norm + 1e-12)
                for p in self.params:
                    if p.grad is None: continue
                    e_w = p.grad * scale
                    p.add_(e_w)
                    self.state[p] = {"e_w": e_w}
                if zero_grad: self.zero_grad()
            def second_step(self, zero_grad=False):
                for p in self.params:
                    if p.grad is None: continue
                    p.sub_(self.state[p]["e_w"])
                self.base_optimizer.step()
                if zero_grad: self.zero_grad()
            def step(self): raise RuntimeError("Use first_step/second_step for SAM")
            def zero_grad(self, set_to_none=True): self.base_optimizer.zero_grad(set_to_none=set_to_none)
            @property
            def param_groups(self): return self.base_optimizer.param_groups
            def state_dict(self): return self.base_optimizer.state_dict()
            def load_state_dict(self, sd): self.base_optimizer.load_state_dict(sd)

        return SimpleSAM(model.parameters(), base_opt_cls, rho=rho, lr=lr, momentum=momentum, weight_decay=weight_decay)

    raise ValueError(f"Unknown optimizer {optim_name}")

def get_scheduler(scheduler_name: str, optimizer, epochs:int, step_size=30, gamma=0.1, patience=5):
    name = scheduler_name.lower()
    opt_for_scheduler = optimizer.optimizers[0] if isinstance(optimizer, CombinedOptimizer) else optimizer

    if name == "steplr":
        return torch.optim.lr_scheduler.StepLR(opt_for_scheduler, step_size=step_size, gamma=gamma)
    if name in ("reducelronplateau","reducelr"):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt_for_scheduler, mode='min', patience=patience, factor=gamma)
    return torch.optim.lr_scheduler.StepLR(opt_for_scheduler, step_size=step_size, gamma=gamma)

def accuracy_from_outputs(outputs, labels):
    _, pred = outputs.max(1)
    return (pred == labels).sum().item()

def evaluate(model, loader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += float(loss.item()) * labels.size(0)
            total_correct += accuracy_from_outputs(outputs, labels)
            total += labels.size(0)
    avg_loss = (total_loss / total) if (criterion is not None and total>0) else 0.0
    acc = 100.0 * total_correct / total if total>0 else 0.0
    return avg_loss, acc

def train_one_epoch(model, loader, optimizer, device, criterion, scaler:Optional[torch.cuda.amp.GradScaler], use_sam=False):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    is_sam = hasattr(optimizer, "first_step") and hasattr(optimizer, "second_step")

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if hasattr(optimizer, "zero_grad"):
            optimizer.zero_grad()
        else:
            try:
                optimizer.zero_grad(set_to_none=True)
            except Exception:
                optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                outputs = model(images)
                if isinstance(outputs, tuple): outputs = outputs[0]
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()

            if is_sam:
                base_for_unscale = optimizer.base_optimizer if hasattr(optimizer, "base_optimizer") else (optimizer.optimizers[0] if isinstance(optimizer, CombinedOptimizer) else optimizer)
                scaler.unscale_(base_for_unscale)
                optimizer.first_step(zero_grad=True)
                with torch.amp.autocast(device_type=device.type):
                    outputs2 = model(images)
                    if isinstance(outputs2, tuple): outputs2 = outputs2[0]
                    loss2 = criterion(outputs2, labels)
                scaler.scale(loss2).backward()
                scaler.unscale_(base_for_unscale)
                optimizer.second_step(zero_grad=True)
                scaler.update()
            else:
                scaler.step(optimizer)
                scaler.update()
        else:
            outputs = model(images)
            if isinstance(outputs, tuple): outputs = outputs[0]
            loss = criterion(outputs, labels)
            loss.backward()
            if is_sam:
                optimizer.first_step(zero_grad=True)
                outputs2 = model(images)
                if isinstance(outputs2, tuple): outputs2 = outputs2[0]
                loss2 = criterion(outputs2, labels)
                loss2.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        running_correct += accuracy_from_outputs(outputs, labels)
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = 100.0 * running_correct / total
    return avg_loss, acc

def default_batch_size_scheduler(epoch, base_batch):
    factor = 1 + min(3, epoch // 20)
    return int(base_batch * factor)

def parse_args():
    p = argparse.ArgumentParser(description="Assignment training pipeline (TensorBoard)")
    p.add_argument("--dataset", type=str, default="cifar-10", choices=["mnist","cifar-10","cifar-100","oxfordiiitpet"])
    p.add_argument("--model", type=str, default="resnet18", choices=["resnet18","resnet50","resnet14d","resnet26d","MLP"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--optimizer", type=str, default="adam", choices=["sgd","adam","adamw","muon","sam"])
    p.add_argument("--scheduler", type=str, default="steplr", choices=["steplr","reducelronplateau"])
    p.add_argument("--augmentations", type=str, nargs="*", default=["crop","flip","colorjitter"])
    p.add_argument("--cache-size", type=int, default=4096)
    p.add_argument("--allow-cpu", action="store_true", help="Allow running on CPU if CUDA not available.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-path", type=str, default="best_model.pth")
    return p.parse_args()

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not torch.cuda.is_available():
        if not args.allow_cpu:
            raise RuntimeError("CUDA is not available but assignment requires CUDA. Use --allow-cpu to permit CPU.")
        device = torch.device("cpu")
        print("WARNING: running on CPU.")
    else:
        device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    train_t = get_transforms(args.model, args.dataset, args.augmentations, is_train=True)
    test_t = get_transforms(args.model, args.dataset, [], is_train=False)

    if args.dataset == "mnist":
        train_ds = MNIST("data", train=True, download=True, transform=train_t)
        test_ds  = MNIST("data", train=False, download=True, transform=test_t)
        num_classes = 10
    elif args.dataset == "cifar-10":
        train_ds = CIFAR10("data", train=True, download=True, transform=train_t)
        test_ds  = CIFAR10("data", train=False, download=True, transform=test_t)
        num_classes = 10
    elif args.dataset == "cifar-100":
        train_ds = CIFAR100("data", train=True, download=True, transform=train_t)
        test_ds  = CIFAR100("data", train=False, download=True, transform=test_t)
        num_classes = 100
    elif args.dataset == "oxfordiiitpet":
        full = OxfordIIITPet("data", split="trainval", target_types="category", download=True, transform=train_t)
        val_size = int(0.1 * len(full))
        train_size = len(full) - val_size
        train_ds, test_ds = random_split(full, [train_size, val_size])
        num_classes = 37
    else:
        raise ValueError("Unknown dataset")

    train_ds = CachedDataset(train_ds, cache_size=args.cache_size)
    test_ds = CachedDataset(test_ds, cache_size=max(128, args.cache_size//8))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=(device.type==device))
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=(device.type==device))

    model = get_model(args.model, num_classes=num_classes, pretrained=True, device=device.type, dataset_name=args.dataset).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = get_optimizer(args.optimizer, model, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = get_scheduler(args.scheduler, optimizer, epochs=args.epochs)

    scaler = torch.amp.GradScaler(device=device.type)

    writer = SummaryWriter(log_dir=os.path.join("first_runs", f"{args.model}_{args.dataset}_{int(time.time())}"))

    best_loss = float("inf")
    best_acc = 0.0
    best_state = None
    no_improve = 0
    early_patience = 7

    print("Starting training:", args.model, "on", args.dataset, "device:", device)
    for epoch in range(args.epochs):
        epoch_start = time.time()

        if epoch > 0:
            new_bs = default_batch_size_scheduler(epoch, args.batch_size)
            if new_bs != train_loader.batch_size:
                train_loader = DataLoader(train_ds, batch_size=new_bs, shuffle=True, num_workers=4, pin_memory=(device.type=="cuda"))
                print(f"Epoch {epoch}: changed train batch size to {new_bs}")

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, criterion, scaler, use_sam=(args.optimizer=="sam"))
        t_train = time.time() - t0

        val_loss, val_acc = evaluate(model, test_loader, device, criterion)
        epoch_time = time.time() - epoch_start

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        try:
            current_lr = optimizer.param_groups[0]['lr']
        except Exception:
            current_lr = optimizer.param_groups[0]['lr'] if len(optimizer.param_groups) > 0 else args.lr
        writer.add_scalar("LR", current_lr, epoch)

        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | Val Loss {val_loss:.4f} Acc {val_acc:.2f}% | Time {epoch_time:.1f}s (train {t_train:.1f}s) | LR {current_lr:.5g}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_acc = val_acc
            best_state = model.state_dict()
            no_improve = 0
            torch.save({"model_state": best_state, "epoch": epoch, "val_loss": best_loss, "val_acc": best_acc}, args.save_path)
        else:
            no_improve += 1
            if no_improve >= early_patience:
                print("Early stopping triggered.")
                break

    writer.close()

    if best_state is not None:
        print(f"Best val loss: {best_loss:.4f}, best val acc: {best_acc:.2f}%. Saved to {args.save_path}")
    else:
        print("No best model saved (training may have failed).")

if __name__ == "__main__":
    main()
