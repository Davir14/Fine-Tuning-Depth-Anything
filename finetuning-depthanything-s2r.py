"""
Fine-tuning di Depth Anything V2 ViT-B Metric Outdoor su TartanAir Stress Test
================================================================================
Struttura attesa del dataset:
    tartanair_stress_test/
    ├── abandonedfactory_night/
    │   ├── Easy/{image_left, depth_left}
    │   └── Hard/{image_left, depth_left}
    ├── carwelding/
    │   ├── Easy/{image_left, depth_left}
    │   └── Hard/{image_left, depth_left}
    └── endofworld/
        ├── Easy/{image_left, depth_left}
        └── Hard/{image_left, depth_left}
"""

import os
import random
import warnings
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from matplotlib.colors import LogNorm
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    DepthAnythingForDepthEstimation,
    get_cosine_schedule_with_warmup,
)

warnings.filterwarnings("ignore", category=UserWarning)

from datetime import datetime

# ============================================================
# 0.  CONFIGURAZIONE GLOBALE
# ============================================================

CFG = {
    # --- Percorsi ---
    "tartanair_dir"       : "tartanair_stress_test",
    "checkpoint_dir" : "checkpoints/cosine-nsaw-go-e-0.1lambdagrad",
    "results_dir"    : "results/cosine-nsaw-go-e-0.1lambdagrad",

    "train_scenes" : {
        "neighborhood",
        "seasonsforest_winter",
        "abandonedfactory_night",
        "westerndesert"
    },

    "val_scenes"   : {
        "gascola",
        "oldtown" 
    },
    
    "test_scenes"  : {
        "endofworld"
    },

    # --- Subsampling video (stride fisso, nessuna posa disponibile) ---
    # Easy: la camera si muove meno → frame più simili → stride più ampio
    # Hard: la camera si muove di più → frame più diversi → stride più stretto
    "stride_easy"    : 5,
    "stride_hard"    : 3,

    # --- Training ---
    "img_size"       : 518,   # risoluzione nativa di Depth Anything
    "batch_size"     : 4,
    "num_workers"    : 2,
    "epochs"         : 10,
    "num_cycles"     : 2,
    "lr_encoder"     : 1e-6,  # LR basso: preserva le feature
    "lr_decoder"     : 5e-6,  # LR più alto per il decoder
    "weight_decay"   : 1e-4,
    "lambda_grad"    : 0.1,   # peso della gradient loss nella loss combinata
    "variance_focus" : 0.85,  # iperparametro della SILog (bilancia scala vs struttura)

    # --- Depth ---
    "max_depth"      : 80.0,  # metri

    # --- Altro ---
    "seed"           : int(datetime.now().timestamp()),
    "device"         : "cuda" if torch.cuda.is_available() else "cpu",
    "viz_samples"    : 4,

    # --- Configurazione del main ---
    "do_training"    : False,
    "do_demo_pretrained": False,
    "do_ensemble"    : True
}

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf"

def set_seed(seed):
    """ imposta il random seed a tutti i moduli che lo richiedono """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 1.  DATASET
# ============================================================

def discover_sequences(tartanair_dir):
    """
    Scansiona tartanair_dir e ritorna una lista di dict, uno per ogni
    combinazione (scena, difficoltà) trovata.

    Struttura attesa:
        tartanair_dir/<scene>/<Easy|Hard>/image_left/*.png
        tartanair_dir/<scene>/<Easy|Hard>/depth_left/*.npy

    Ogni dict contiene:
        scene      : str  es. 'abandonedfactory_night'
        difficulty : str  es. 'Easy' | 'Hard'
        images     : [path, ...]  allineati con depths per stem
        depths     : [path, ...]
    """
    sequences = []
    root = Path(tartanair_dir)

    for img_dir in sorted(root.rglob("image_left")):
        depth_dir = img_dir.parent / "depth_left"
        if not depth_dir.exists():
            continue

        images = sorted(img_dir.glob("*.png"))
        depths = sorted(depth_dir.glob("*.npy"))

        # Allineamento per stem:
        # image = 000000_left.png  →  stem = '000000_left'
        # depth = 000000_left_depth.npy  →  stem senza '_depth' = '000000_left'
        depth_stems = {d.stem.replace("_depth", ""): d for d in depths}
        aligned_imgs, aligned_deps = [], []
        for img in images:
            key = img.stem
            if key in depth_stems:
                aligned_imgs.append(str(img))
                aligned_deps.append(str(depth_stems[key]))
            else:
                print(f"[WARN] Nessuna depth per {img.name} — saltato")

        if not aligned_imgs:
            print(f"[WARN] Nessun frame allineato in {img_dir} — saltato")
            continue

        # img_dir = .../abandonedfactory_night/Hard/image_left
        # .parent = .../abandonedfactory_night/Hard      -> difficulty
        # .parent.parent = .../abandonedfactory_night    -> scene
        difficulty = img_dir.parent.name
        scene      = img_dir.parent.parent.name

        sequences.append({
            "scene"     : scene,
            "difficulty": difficulty,
            "images"    : aligned_imgs,
            "depths"    : aligned_deps,
        })
        print(f"[INFO] '{scene}/{difficulty}': {len(aligned_imgs)} frame allineati")

    return sequences


def build_index(sequences, stride_easy=5, stride_hard=3):
    """
    Applica stride fisso differenziato per difficoltà.
    Ritorna lista di (img_path, depth_path, scene, difficulty).
    """
    index = []
    for seq in sequences:
        stride = stride_hard if seq["difficulty"] == "Hard" else stride_easy
        sel    = list(range(0, len(seq["images"]), stride))
        
        print(f"[INFO] '{seq['scene']}/{seq['difficulty']}': "
              f"{len(sel)}/{len(seq['images'])} frame selezionati (stride={stride})")
        
        for i in sel:
            index.append((
                seq["images"][i],
                seq["depths"][i],
                seq["scene"],
                seq["difficulty"],
            ))
    return index


def split_index(index, train_scenes, val_scenes, test_scenes):
    """
    Divide l'indice in train / val / test per nome scena.
    Tutte le difficoltà (Easy + Hard) di una scena vanno nello stesso bucket.
    """
    train, val, test = [], [], []
    for item in index:
        _, _, scene, diff = item
        if scene in train_scenes:
            train.append(item)
        elif scene in val_scenes:
            val.append(item)
        elif scene in test_scenes:
            test.append(item)
    return train, val, test


# ---- Transforms ----


def get_train_transform(img_size):
    return A.Compose([
        # --- GEOMETRICHE ---
        # Padding per arrivare almeno alla risoluzione richiesta
        A.PadIfNeeded(
            min_height=img_size, min_width=img_size, 
            border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0.0
        ),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.0, rotate_limit=5, p=0.3,
            border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0.0
        ),
        # Crop casuale della dimensione esatta (senza resizing/zoom)
        A.RandomCrop(height=img_size, width=img_size, p=1.0),
        A.HorizontalFlip(p=0.5),

        # --- PIXEL-LEVEL (solo RGB, ignorate dalla mask) ---
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2, hue=0.05, p=0.8),
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=0.1),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.ISONoise(p=1.0),
            #A.ImageCompression(quality_lower=70, quality_upper=90, p=1.0),
        ], p=0.3),
        
        # Normmalizzazione ImageNet atteso da depth anything
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ], additional_targets={"depth": "mask"}) # applica le trasformazioni anche alla depth, ma considerata come mask


def get_val_transform(img_size):
    return A.Compose([
        # Padding per arrivare almeno alla risoluzione richiesta
        A.PadIfNeeded(
            min_height=img_size, min_width=img_size, 
            border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0.0
        ),
        
        # Crop centrale
        A.CenterCrop(height=img_size, width=img_size, p=1.0),
        
        # Normmalizzazione ImageNet atteso da depth anything
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ], additional_targets={"depth": "mask"})


class TartanAirDataset(Dataset):
    def __init__(self, index, transform=None):
        if not index:
            raise ValueError("Index vuoto.")
        self.index     = index
        self.transform = transform

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        img_path, depth_path, scene, difficulty = self.index[idx]

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path).astype(np.float32)

        #depth = np.where(~np.isfinite(depth) | (depth < 0) | (depth > CFG["max_depth"]), np.nan, depth)
        depth = np.clip(depth, 0.0, CFG["max_depth"])

        if self.transform:
            aug = self.transform(image=image, depth=depth)
            
            # 1. Estrazione degli array NumPy
            image_np = aug["image"]
            depth_np = aug["depth"]
            
            # 2. Conversione esplicita e sicura in Tensor
            # RGB: Trasponiamo HWC -> CHW
            image = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
            
            # Depth: Già (H, W), basta il cast
            depth = torch.from_numpy(depth_np).float()
        else:
            image_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mean    = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std     = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image   = (image_t - mean) / std
            depth   = torch.from_numpy(depth).float()

        return image, depth, f"{scene}/{difficulty}"

# ============================================================
# 2.  MODELLO
# ============================================================

def build_model(model_id=MODEL_ID, device="cpu", num_layers_to_freeze=8):
    """ Carica il modello con i layer congelati richiesti """
    print(f"[INFO] Caricamento modello: {model_id}")
    model     = DepthAnythingForDepthEstimation.from_pretrained(model_id)
    model     = model.to(device)
    total     = sum(p.numel() for p in model.parameters())

    freeze_encoder_layers(model, num_layers_to_freeze=num_layers_to_freeze)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parametri totali:    {total:,}")
    print(f"[INFO] Parametri trainable: {trainable:,}")
    return model

def freeze_encoder_layers(model, num_layers_to_freeze=8):
    """ Congela i primi N layer del backbone ViT-B (12 totali). """
    frozen = 0
    for i, layer in enumerate(model.backbone.encoder.layer):
        if i < num_layers_to_freeze:
            for p in layer.parameters():
                p.requires_grad = False
            frozen += 1
    print(f"[INFO] Congelati {frozen}/12 layer del backbone")


# ============================================================
# 3.  LOSS
# ============================================================

def silog_loss(pred, gt, valid_mask, variance_focus=0.85):
    p = pred[valid_mask].clamp(min=1e-3)
    g = gt[valid_mask].clamp(min=1e-3)
    d = torch.log(p) - torch.log(g)
    return torch.sqrt((d ** 2).mean() - variance_focus * (d.mean() ** 2) + 1e-6)

def gradient_loss(pred, gt, valid_mask):
    if pred.ndim == 3:
        pred       = pred.unsqueeze(1)
        gt         = gt.unsqueeze(1)
        valid_mask = valid_mask.unsqueeze(1)
    
    # B x C x H x W
    pred_dx = pred[:, :, :, 1:]  - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :]  - pred[:, :, :-1, :]
    gt_dx   = gt[:, :, :, 1:]    - gt[:, :, :, :-1]
    gt_dy   = gt[:, :, 1:, :]    - gt[:, :, :-1, :]

    mask_x  = valid_mask[:, :, :, 1:] & valid_mask[:, :, :, :-1]
    mask_y  = valid_mask[:, :, 1:, :] & valid_mask[:, :, :-1, :]

    parts = []
    lx = torch.abs(pred_dx[mask_x] - gt_dx[mask_x])
    ly = torch.abs(pred_dy[mask_y] - gt_dy[mask_y])
    
    if lx.numel() > 0: parts.append(lx.mean())
    if ly.numel() > 0: parts.append(ly.mean())
    if not parts:
        return torch.tensor(0.0, device=pred.device)
    return sum(parts) / len(parts)
    
def combined_loss(pred, gt, valid_mask, lambda_grad=0.5, variance_focus=0.85):
    si   = silog_loss(pred, gt, valid_mask, variance_focus)
    grad = gradient_loss(pred, gt, valid_mask)
    return si + lambda_grad * grad, si.item(), grad.item()

# ============================================================
# 4.  METRICHE
# ============================================================

@torch.no_grad()
def compute_metrics(pred, gt, valid_mask):
    """Calcola le metriche AbsRel, RMSE (m) e delta1 (soglia 1.25)"""
    p = pred[valid_mask].clamp(min=1e-3)
    g = gt[valid_mask].clamp(min=1e-3)
    abs_rel = ((p - g).abs() / g).mean().item()
    rmse    = torch.sqrt(((p - g) ** 2).mean()).item()
    delta1  = (torch.max(p / g, g / p) < 1.25).float().mean().item()
    return {"AbsRel": abs_rel, "RMSE": rmse, "delta1": delta1}


# ============================================================
# 5.  INFERENCE HELPER
# ============================================================

def forward_model(model, images):
    """
    Wrapper del modello di depth.
    Input : (B, 3, H, W) normalizzato
    Output: (B, H, W) in metri [0, 80]
    """
    return model(pixel_values=images).predicted_depth


# ============================================================
# 6.  DEMO PRE-TRAINING
# ============================================================

@torch.no_grad()
def demo_pretrained(model, dataset, device, save_path, imgs_to_show=20):
    """
    Visualizza campioni casuali del dataset per visualizzare il modello prima del finetuning
    """
    model.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 1. Setup figure
    num_samples = min(imgs_to_show, len(dataset))
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1: axes = [axes]

    # Costanti per denormalizzare le immagini (inversa di Normalize)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    indices = random.sample(range(len(dataset)), k=num_samples)
    print(f"DEBUG: Immagini scelte (indici): {indices}")

    for row, idx in enumerate(indices):
        # 2. Recupero dati
        image_tensor, depth_gt, label = dataset[idx]

        pred = model(image_tensor.unsqueeze(0).to(device)).predicted_depth.squeeze(0).cpu()

        # 3. Denormalizza per la visualizzazione
        img_vis = (image_tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        # 4. Identifica i pixel validi
        valid_mask = torch.isfinite(depth_gt) & (depth_gt > 0)
        gt_vis = depth_gt.numpy().copy()
        
        # Porta a NaN i pixel invalidi qualora ci siano in modo da non visualizzarli
        pred[~valid_mask.numpy()] = np.nan
        gt_vis[~valid_mask.numpy()] = np.nan

        # 5. Plotting
        # RGB
        axes[row][0].imshow(img_vis)
        axes[row][0].set_title(f"RGB — {label}")
        axes[row][0].axis("off")

        # Debug prima del calcolo dei min/max
        print(f"DEBUG: Label={label}")
        print(f"DEBUG: depth_gt shape={depth_gt.shape}, min={depth_gt.min()}, max={depth_gt.max()}")
        print(f"DEBUG: Pixel validi trovati={valid_mask.sum().item()} su {valid_mask.numel()}")
        # Calcola min e max di questa immagine
        vmin_val = np.nanmin(gt_vis)
        vmax_val = np.nanmax(gt_vis)

        # log norm
        vmin_log = max(0.1, vmin_val)
        norm = LogNorm(vmin=vmin_log, vmax=vmax_val)

        # Ground Truth con scala logaritmica
        im1 = axes[row][1].imshow(gt_vis, cmap="plasma", norm=norm)
        axes[row][1].set_title("Ground Truth (m) - Log Scale")
        axes[row][1].axis("off")
        plt.colorbar(im1, ax=axes[row][1], fraction=0.046, pad=0.04)

        # Predizione con la stessa normalizzazione per confronto diretto
        im2 = axes[row][2].imshow(pred.numpy(), cmap="plasma", norm=norm)
        axes[row][2].set_title("Pred pre-FT (m) - Log Scale")
        axes[row][2].axis("off")
        plt.colorbar(im2, ax=axes[row][2], fraction=0.046, pad=0.04)

    plt.suptitle("Verifica Pre-Training (NaN gestiti correttamente)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[INFO] Demo pre-training salvata in: {save_path}")

# ============================================================
# 7.  TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(model, loader, scheduler, optimizer, device):
    model.train()
    total_loss = total_si = total_grad = 0.0
    all_metrics = {"AbsRel": [], "RMSE": [], "delta1": []}

    pbar = tqdm(loader, desc="  Train", leave=False)
    for batch_idx, (images, depths, _) in enumerate(pbar):
        # Azzera i gradienti all'inizio di ogni iterazione
        optimizer.zero_grad()

        images = images.to(device)
        depths = depths.to(device)
        valid_mask = torch.isfinite(depths) & (depths > 0)

        pred = forward_model(model, images)

        if pred.shape != depths.shape:
            pred = nn.functional.interpolate(
                pred.unsqueeze(1), size=depths.shape[-2:],
                mode="bilinear", align_corners=False
            ).squeeze(1)

        loss, si, grad = combined_loss(
            pred, depths, valid_mask,
            lambda_grad=CFG["lambda_grad"],
            variance_focus=CFG["variance_focus"],
        )

        # Backpropagation
        loss.backward()

        # Ottimizzazione ad ogni singolo batch
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # logging
        loss_display = loss.item()
        m = compute_metrics(pred.detach(), depths, valid_mask)
        for k in all_metrics:
            all_metrics[k].append(m[k])
        total_loss += loss_display
        total_si   += si
        total_grad += grad

        pbar.set_postfix({"loss": f"{loss_display:.4f}",
                          "AbsRel": f"{m['AbsRel']:.4f}"})

    n = len(loader)
    return {
        "loss"  : total_loss / n,
        "silog" : total_si / n,
        "grad"  : total_grad / n,
        "AbsRel": np.mean(all_metrics["AbsRel"]),
        "RMSE"  : np.mean(all_metrics["RMSE"]),
        "delta1": np.mean(all_metrics["delta1"]),
    }

# ============================================================
# 8.  EVALUATE
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_metrics = {"AbsRel": [], "RMSE": [], "delta1": []}

    for images, depths, _ in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device)
        depths = depths.to(device)
        valid_mask = torch.isfinite(depths) & (depths > 0)

        pred = forward_model(model, images)
        if pred.shape != depths.shape:
            pred = nn.functional.interpolate(
                pred.unsqueeze(1), size=depths.shape[-2:],
                mode="bilinear", align_corners=False
            ).squeeze(1)

        loss, _, _ = combined_loss(
            pred, depths, valid_mask,
            lambda_grad=CFG["lambda_grad"],
            variance_focus=CFG["variance_focus"],
        )
        total_loss += loss.item()

        m = compute_metrics(pred, depths, valid_mask)
        for k in all_metrics:
            all_metrics[k].append(m[k])

    return {
        "loss"  : total_loss / len(loader),
        "AbsRel": np.mean(all_metrics["AbsRel"]),
        "RMSE"  : np.mean(all_metrics["RMSE"]),
        "delta1": np.mean(all_metrics["delta1"]),
    }


# ============================================================
# 9.  TRAINING LOOP
# ============================================================

from transformers import  get_cosine_schedule_with_warmup
import os
import torch

# schedulers

def sched_cosine_warmup(optimizer, train_loader):
    # steps effettivi per epoca = batch totali
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * CFG["epochs"]
    
    warmup_epochs = 3
    warmup_steps = steps_per_epoch * warmup_epochs
    
    print(f"[INFO] Scheduler: Cosine with Warmup | Total steps: {total_steps} | Warmup steps: {warmup_steps}")

    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

# training
def train(model, train_loader, val_loader, device):
    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)

    encoder_params = list(model.backbone.parameters())
    decoder_params = [p for n, p in model.named_parameters()
                      if "backbone" not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": [p for p in encoder_params if p.requires_grad],
         "lr": CFG["lr_encoder"]},
        {"params": decoder_params,
         "lr": CFG["lr_decoder"]},
    ], weight_decay=CFG["weight_decay"])

    scheduler = sched_cosine_warmup(optimizer, train_loader)

    history    = {"train_loss": [], "val_loss": [],
                  "val_AbsRel": [], "val_RMSE": [], "val_delta1": []}
    best_absrel = float("inf")

    print(f"\n{'='*60}")
    print(f"  Fine-tuning  |  epochs={CFG['epochs']}  |  device={device}")
    print(f"{'='*60}\n")

    for epoch in range(1, CFG["epochs"] + 1):
        print(f"Epoch {epoch:03d}/{CFG['epochs']:03d}")

        train_stats = train_one_epoch(model, train_loader, scheduler, optimizer, device)
        val_stats   = evaluate(model, val_loader, device)

        print(f"  Train → loss={train_stats['loss']:.4f}  "
              f"SILog={train_stats['silog']:.4f}  Grad={train_stats['grad']:.4f}  "
              f"AbsRel={train_stats['AbsRel']:.4f}")
        print(f"  Val   → loss={val_stats['loss']:.4f}  "
              f"AbsRel={val_stats['AbsRel']:.4f}  "
              f"RMSE={val_stats['RMSE']:.4f}  δ1={val_stats['delta1']:.4f}")

        for k, v in [("train_loss", train_stats["loss"]),
                     ("val_loss",   val_stats["loss"]),
                     ("val_AbsRel", val_stats["AbsRel"]),
                     ("val_RMSE",   val_stats["RMSE"]),
                     ("val_delta1", val_stats["delta1"])]:
            history[k].append(v)

        if val_stats["AbsRel"] < best_absrel:
            best_absrel = val_stats["AbsRel"]
            torch.save({
                "epoch"     : epoch,
                "state_dict": model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "val_AbsRel": best_absrel,
            }, os.path.join(CFG["checkpoint_dir"], "best_model.pth"))
            print(f"    Best checkpoint salvato (AbsRel={best_absrel:.4f})")

        print()

    torch.save({"epoch": CFG["epochs"], "state_dict": model.state_dict()},
               os.path.join(CFG["checkpoint_dir"], "last_model.pth"))
    
    return history

# ============================================================
# 10.  PLOT HISTORY
# ============================================================

# plotta l'andamento delle loss e delle metriche per ogni epoca
def plot_history(history, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train", lw=2)
    axes[0].plot(epochs, history["val_loss"],   label="Val",   lw=2)
    axes[0].set_title("Loss (SILog + Grad)")
    axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.4)

    axes[1].plot(epochs, history["val_AbsRel"], color="tomato", lw=2)
    axes[1].set_title("Val AbsRel ↓")
    axes[1].set_xlabel("Epoch"); axes[1].grid(True, alpha=0.4)

    axes[2].plot(epochs, history["val_delta1"], color="steelblue", lw=2)
    axes[2].set_title("Val δ1 ↑  (thresh 1.25)")
    axes[2].set_xlabel("Epoch"); axes[2].grid(True, alpha=0.4)

    plt.suptitle("Training History — Depth Anything V2 ViT-B FT",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[INFO] History plot salvato in: {save_path}")


# ============================================================
# 11.  TEST + VISUALIZZAZIONE QUALITATIVA
# ============================================================

@torch.no_grad()
def test(model, test_loader, device):
    """Valutazione quantitativa sul test set."""
    stats = evaluate(model, test_loader, device)
    print(f"\n{'='*50}")
    print(f"  TEST RESULTS  ({CFG["test_scenes"]})")
    print(f"{'='*50}")
    print(f"  AbsRel : {stats['AbsRel']:.4f}")
    print(f"  RMSE   : {stats['RMSE']:.4f} m")
    print(f"  δ1     : {stats['delta1']:.4f}  (ideale → 1.0)")
    print(f"{'='*50}\n")
    return stats


@torch.no_grad()
def visualize_comparison(base_model, ft_model, dataset, device, save_path):
    """
    Esegue inferenza con i due modelli su alcune immagini del dataset
    """
    base_model.eval()
    ft_model.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    num_samples = min(8, len(dataset))
    # 4 colonne: RGB | Ground Truth | Pre-Trained | Fine-Tuned
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    if num_samples == 1: axes = [axes]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    indices = random.sample(range(len(dataset)), k=num_samples)
    print(f"[INFO] Generazione confronto sui frame: {indices}")

    for row, idx in enumerate(indices):
        image_tensor, depth_gt, label = dataset[idx]
        img_input = image_tensor.unsqueeze(0).to(device)

        # Inferenza parallela sui due modelli
        pred_base = forward_model(base_model, img_input).squeeze(0).cpu()
        pred_ft   = forward_model(ft_model, img_input).squeeze(0).cpu()

        # Denormalizzazione RGB
        img_vis = (image_tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        # Mascheratura GT
        valid_mask = torch.isfinite(depth_gt) & (depth_gt > 0)
        gt_vis = depth_gt.numpy().copy()
        pred_base[~valid_mask.numpy()] = np.nan
        pred_ft[~valid_mask.numpy()] = np.nan
        gt_vis[~valid_mask.numpy()] = np.nan

        # Setup per la normalizzazione del colore per la visualizzazione
        vmin_val = np.nanmin(gt_vis)
        vmax_val = np.nanmax(gt_vis)
        vmin_log = max(0.1, vmin_val)
        norm = LogNorm(vmin=vmin_log, vmax=vmax_val)

        # Plot RGB
        axes[row][0].imshow(img_vis)
        axes[row][0].set_title(f"RGB — {label}")
        axes[row][0].axis("off")

        # Plot GT
        im1 = axes[row][1].imshow(gt_vis, cmap="plasma", norm=norm)
        axes[row][1].set_title("Ground Truth (m)")
        axes[row][1].axis("off")
        plt.colorbar(im1, ax=axes[row][1], fraction=0.046, pad=0.04)

        # Plot Base (Pre-Trained)
        im2 = axes[row][2].imshow(pred_base.numpy(), cmap="plasma", norm=norm)
        axes[row][2].set_title("Base Model (Metric)")
        axes[row][2].axis("off")
        plt.colorbar(im2, ax=axes[row][2], fraction=0.046, pad=0.04)

        # Plot Fine-Tuned
        im3 = axes[row][3].imshow(pred_ft.numpy(), cmap="plasma", norm=norm)
        axes[row][3].set_title("Fine-Tuned Model")
        axes[row][3].axis("off")
        plt.colorbar(im3, ax=axes[row][3], fraction=0.046, pad=0.04)

    plt.suptitle("Confronto Qualitativo: Pre-Trained vs Fine-Tuned", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[DONE] Immagine comparativa salvata in: {save_path}")

# ============================================================
# 12.  ENSEMBLE
# ============================================================

@torch.no_grad()
def ensemble_prediction(model, checkpoint_paths, image_tensor, device):
    """
    Esegue un ensemble iterativo caricando i pesi sequenzialmente su un singolo modello.
    Returns: mean_depth, std_depth, confidence (numpy arrays)
    """
    image_tensor = image_tensor.to(device)
    predictions = []

    for ckpt_path in checkpoint_paths:
        # Carica i pesi dal disco
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        # Inferenza
        pred = model(pixel_values=image_tensor).predicted_depth.squeeze(0).cpu()
        predictions.append(pred)

    # Stack: tensore di shape (N, H, W)
    stacked_preds = torch.stack(predictions, dim=0)

    # Statistiche pixel-wise
    mean_depth = torch.mean(stacked_preds, dim=0)
    std_depth = torch.std(stacked_preds, dim=0, unbiased=True) # Incertezza (metri)
    confidence = torch.exp(-std_depth)                         # Confidenza [0, 1]

    return mean_depth.numpy(), std_depth.numpy(), confidence.numpy()

@torch.no_grad()
def visualize_ensemble_dataset(model, checkpoint_paths, dataset, device, save_path, num_samples=4):
    """
    Estrae N campioni dal dataset, applica l'ensemble e genera un plot con:
    RGB | Ground Truth | Depth Media | Incertezza (Std) | Confidenza
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    indices = random.sample(range(len(dataset)), k=min(num_samples, len(dataset)))
    print(f"[INFO] Generazione Ensemble visuale sui frame: {indices}")

    # 5 Colonne per visualizzazione
    fig = plt.figure(figsize=(25, 5 * len(indices)))
    gs = gridspec.GridSpec(len(indices), 5, figure=fig, hspace=0.3, wspace=0.05)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for row, idx in enumerate(indices):
        # Estrazione dal Dataset
        image_tensor, depth_gt, label = dataset[idx]
        img_input = image_tensor.unsqueeze(0) # Shape: (1, 3, H, W)

        # Chiamata all'Ensemble
        mean_depth, std_depth, conf_map = ensemble_prediction(
            model, checkpoint_paths, img_input, device
        )

        # Denormalizzazione per visualizzazione
        img_vis = (image_tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        # Gestione Maschera GT
        valid_mask = torch.isfinite(depth_gt) & (depth_gt > 0)
        gt_vis = depth_gt.numpy().copy()
        
        gt_vis[~valid_mask.numpy()] = np.nan
        mean_depth[~valid_mask.numpy()] = np.nan
        std_depth[~valid_mask.numpy()]  = np.nan
        conf_map[~valid_mask.numpy()]   = np.nan

        # normalizzazione
        vmin_val = np.nanmin(gt_vis)
        vmax_val = max(np.nanmax(gt_vis), 1.0)
        norm_depth = LogNorm(vmin=max(0.1, vmin_val), vmax=vmax_val)

        # Per l'incertezza, limitiamo il vmax al 95° percentile per non farla appiattire da outlier
        vmax_std = np.nanpercentile(std_depth, 95) if not np.isnan(std_depth).all() else 1.0

        # 5. Plotting (Colonna per Colonna)
        columns = [
            (img_vis,    f"RGB — {label}", None, None, None),
            (gt_vis,     "Ground Truth (m)", "plasma", norm_depth, None),
            (mean_depth, "Ensemble Mean (m)", "plasma", norm_depth, None),
            (std_depth,  "Uncertainty (Std Dev)", "inferno", None, (0, vmax_std)),
            (conf_map,   "Confidence Map (0 to 1)", "viridis", None, (0, 1.0))
        ]

        for col, (data, title, cmap, norm, vlim) in enumerate(columns):
            ax = fig.add_subplot(gs[row, col])
            if cmap is None:
                ax.imshow(data)
            else:
                vmin_, vmax_ = vlim if vlim else (None, None)
                im = ax.imshow(data, cmap=cmap, norm=norm, vmin=vmin_, vmax=vmax_)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title, fontsize=11)
            ax.axis("off")

    plt.suptitle(f"Deep Ensemble ({len(checkpoint_paths)} Modelli) — Analisi Incertezza",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[DONE] Plot Ensemble salvato in: {save_path}")

# ============================================================
# 13.  MAIN
# ============================================================

def build_dataset():
    print("=== Scoperta sequenze ===")
    sequences  = discover_sequences(CFG["tartanair_dir"])
    full_index = build_index(sequences,
                             stride_easy=CFG["stride_easy"],
                             stride_hard=CFG["stride_hard"])
    train_idx, val_idx, test_idx = split_index(
        full_index,
        train_scenes = CFG["train_scenes"],
        val_scenes   = CFG["val_scenes"],
        test_scenes  = CFG["test_scenes"],
    )
    print(f"\n[INFO] Split -> Train: {len(train_idx)} | "
          f"Val: {len(val_idx)} ({CFG['val_scenes']}) | "
          f"Test: {len(test_idx)} ({CFG['test_scenes']})\n")

    img_size = CFG["img_size"]
    train_ds = TartanAirDataset(train_idx, transform=get_train_transform(img_size))
    val_ds   = TartanAirDataset(val_idx,   transform=get_val_transform(img_size))
    test_ds  = TartanAirDataset(test_idx,  transform=get_val_transform(img_size))

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True,  num_workers=CFG["num_workers"],
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=CFG["num_workers"],
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=CFG["num_workers"],
                              pin_memory=True)
                              
    return val_ds, test_ds, train_loader, val_loader, test_loader

def main():
    set_seed(CFG["seed"])
    device = CFG["device"]
    os.makedirs(CFG["results_dir"], exist_ok=True)
    print(f"[INFO] Device: {device}\n")

    # --- Dataset ---
    val_ds, test_ds, train_loader, val_loader, test_loader = build_dataset()

    # --- Modello ---
    print("=== Caricamento modello ===")
    model = build_model(MODEL_ID, device, num_layers_to_freeze=8)

    if CFG.get("do_demo_pretrained", False):
        # --- Inferenza su alcune immagini prima del fine-tuning ---
        print("\n=== Demo pre-training ===")
        demo_pretrained(model, val_ds, device,
                        save_path=os.path.join(CFG["results_dir"], "demo_pretrained.png"),
                        imgs_to_show=40
        )

    if CFG.get("do_training", False):
        # --- Fine-tuning ---
        print("\n=== Fine-tuning ===")
        history = train(model, train_loader, val_loader, device)
        plot_history(history,
                     save_path=os.path.join(CFG["results_dir"], "training_history.png"))
    
        # --- Carica il best checkpoint ---
        print("\n=== Caricamento best checkpoint ===")
        ckpt = torch.load(os.path.join(CFG["checkpoint_dir"], "best_model.pth"),
                        map_location=device,
                        weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[INFO] Best epoch={ckpt['epoch']}  Val AbsRel={ckpt['val_AbsRel']:.4f}")

        # --- Test quantitativo ---
        print("\n=== Test ===")
        test(model, test_loader, device)

        # --- Creazione Modello Base per Confronto ---
        print("\n=== Istanziazione Modello Base (Pre-trained) ===")
        # build_model caricherà i pesi originali di HuggingFace
        base_model = build_model(MODEL_ID, device)
        
        # --- Visualizzazione qualitativa Comparativa ---
        visualize_comparison(
            base_model=base_model,
            ft_model=model, 
            dataset=test_ds, 
            device=device, 
            save_path=os.path.join(CFG["results_dir"], "test_comparison.png")
        )
    
    if CFG.get("do_ensemble", False):
        print("\n=== Inizializzazione Deep Ensemble ===")
        model = build_model(MODEL_ID, device)
        
        ensemble_checkpoints = [
            "checkpoints/fixedlr-nsaw-go-e/best_model.pth",
            "checkpoints/cosine-nsaw-go-e/best_model.pth",
            "checkpoints/cosine-2cycles-nsaw-go-e/best_model.pth",
            "checkpoints/cosine-nsaw-go-e-0.1lambdagrad/best_model.pth",
            "checkpoints/cosine-nsaw-go-e-0.1lambdagrad-warmup3/best_model.pth"
        ]
        
        visualize_ensemble_dataset(
            model=model,
            checkpoint_paths=ensemble_checkpoints,
            dataset=test_ds,
            device=device,
            save_path=os.path.join(CFG["results_dir"], "ensemble_analysis.png"),
            num_samples=5
        )

    print("\n[DONE] Output salvati in:", CFG["results_dir"])


if __name__ == "__main__":
    main()