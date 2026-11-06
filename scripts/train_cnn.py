"""Entrena la CNN pequeña desde cero (buena vs mala) sobre los recortes 128x128.

- **Muestreo balanceado:** en cada época se toman ``--epoch-size`` imágenes de
  train con probabilidad proporcional a 1/N_estrato (WeightedRandomSampler),
  de modo que cada estrato fruta×calidad aparece con la misma frecuencia
  esperada. Así se compensa el exceso de granada buena sin pesar la pérdida.
- **Aumento de datos por lote** (solo en train): las 8 simetrías del cuadrado
  (rotaciones de 90° y volteos; una fruta no tiene "arriba") y brillo y
  contraste leves. NO se altera el tono ni la saturación, porque el color es
  la señal de calidad. Hacerlo por lote y no imagen por imagen lo vuelve
  ~5 veces más rápido en CPU.
- AdamW + decaimiento coseno de la tasa de aprendizaje.
- Parada temprana con el F1-macro de validación; se guarda el mejor modelo.
- Al final evalúa el mejor modelo en val y test y guarda las probabilidades.

Salidas (con --tag NOMBRE):
    models/cnn_NOMBRE.pt
    reports/cnn/NOMBRE_historia.csv
    reports/cnn/NOMBRE_predicciones.csv
    reports/cnn/NOMBRE_resumen.json

Uso:
    python scripts/train_cnn.py --tag base
    python scripts/train_cnn.py --tag lr3e-4 --lr 3e-4 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fruitq.cnn import MEAN, STD, TILE, SmallCNN, n_params  # noqa: E402
from fruitq.config import MODELS, PROCESSED, SEED  # noqa: E402

OUT = ROOT / "reports" / "cnn"
_MEAN = torch.tensor(MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(STD).view(1, 3, 1, 1)


def cargar():
    m = pd.read_csv(PROCESSED / "manifest.csv")
    tiles = np.load(PROCESSED / f"tiles{TILE}.npy", mmap_mode="r")
    idx = pd.read_csv(PROCESSED / f"tiles{TILE}_index.csv")
    assert (idx.path.values == m.path.values).all(), "tiles y manifest desalineados"
    m["y"] = (m.calidad == "mala").astype(int)
    return m, tiles


def a_tensor(tiles, df) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(np.ascontiguousarray(tiles[df.index.values])).permute(0, 3, 1, 2).contiguous()
    return x, torch.tensor(df.y.values, dtype=torch.long)


def normalizar(x_u8: torch.Tensor) -> torch.Tensor:
    return (x_u8.float() / 255.0 - _MEAN) / _STD


def aumentar(x_u8: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Simetrías del cuadrado + brillo/contraste, vectorizado sobre el lote."""
    x = x_u8.float() / 255.0
    n = x.shape[0]
    k = torch.randint(0, 4, (n,), generator=g)
    flip = torch.rand(n, generator=g) < 0.5
    out = torch.empty_like(x)
    for r in range(4):
        sel = k == r
        if sel.any():
            out[sel] = torch.rot90(x[sel], r, dims=(2, 3))
    out[flip] = out[flip].flip(3)
    mask = (x_u8.sum(1, keepdim=True) > 0).float()  # no "iluminar" el fondo negro
    mask = torch.stack([torch.rot90(mask[i], int(k[i]), dims=(1, 2)) for i in range(n)])
    mask[flip] = mask[flip].flip(3)
    brillo = 1 + (torch.rand(n, 1, 1, 1, generator=g) - 0.5) * 0.4      # ±20 %
    contraste = 1 + (torch.rand(n, 1, 1, 1, generator=g) - 0.5) * 0.4
    media = (out * mask).sum((1, 2, 3), keepdim=True) / (3 * mask.sum((1, 2, 3), keepdim=True) + 1e-6)
    out = ((out - media) * contraste + media) * brillo
    out = (out.clamp(0, 1) * mask)
    return (out - _MEAN) / _STD


@torch.no_grad()
def predecir(model, x_u8, y, batch=256):
    model.eval()
    probs, loss_sum = [], 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for i in range(0, len(y), batch):
        z = model(normalizar(x_u8[i:i + batch]).contiguous(memory_format=torch.channels_last))
        loss_sum += ce(z, y[i:i + batch]).item()
        probs.append(torch.softmax(z, 1)[:, 1].numpy())
    return np.concatenate(probs), loss_sum / len(y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--epoch-size", type=int, default=6000, help="Imágenes muestreadas por época")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--widths", default="16,32,64")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4, help="Núcleos físicos; más hilos no acelera en CPU")
    ap.add_argument("--max-train", type=int, default=0, help="Limita train (para pruebas rápidas)")
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.set_num_threads(args.threads)
    g = torch.Generator().manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True); MODELS.mkdir(exist_ok=True)

    m, tiles = cargar()
    tr, va, te = (m[m.split == s] for s in ("train", "val", "test"))
    if args.max_train:
        tr = tr.sample(args.max_train, random_state=SEED)
        va = va.sample(min(len(va), args.max_train // 4), random_state=SEED)
    x_tr, y_tr = a_tensor(tiles, tr)
    x_va, y_va = a_tensor(tiles, va)
    x_te, y_te = a_tensor(tiles, te)

    n_est = tr.estrato.value_counts()
    w = torch.tensor((1.0 / tr.estrato.map(n_est)).values, dtype=torch.double)
    epoch_size = min(args.epoch_size, len(tr)) if args.max_train else args.epoch_size

    widths = tuple(int(x) for x in args.widths.split(","))
    # channels_last (NHWC) acelera ~1.6x las convoluciones en CPU
    model = SmallCNN(2, widths, args.dropout).to(memory_format=torch.channels_last)
    print(f"SmallCNN{widths}: {n_params(model):,} parámetros | train {len(tr)} (época = {epoch_size} muestras balanceadas)"
          f" | val {len(va)} | test {len(te)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()

    hist, best_f1, best_ep, sin_mejora = [], -1.0, -1, 0
    ckpt = MODELS / f"cnn_{args.tag}.pt"
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time()
        idx = torch.multinomial(w, epoch_size, replacement=True, generator=g)
        tot, aciertos = 0.0, 0
        for i in range(0, epoch_size, args.batch):
            b = idx[i:i + args.batch]
            x, y = aumentar(x_tr[b], g).contiguous(memory_format=torch.channels_last), y_tr[b]
            z = model(x)
            loss = ce(z, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b); aciertos += (z.argmax(1) == y).sum().item()
        sched.step()
        p_va, loss_va = predecir(model, x_va, y_va)
        f1_va = f1_score(y_va.numpy(), p_va >= 0.5, average="macro")
        fila = {"epoca": ep, "loss_train": tot / epoch_size, "acc_train": aciertos / epoch_size, "loss_val": loss_va,
                "acc_val": float(((p_va >= 0.5) == y_va.numpy()).mean()), "f1_val": f1_va,
                "lr": opt.param_groups[0]["lr"], "seg": time.time() - t0}
        hist.append(fila)
        mejora = f1_va > best_f1
        if mejora:
            best_f1, best_ep, sin_mejora = f1_va, ep, 0
            torch.save({"state_dict": model.state_dict(), "widths": widths, "dropout": args.dropout,
                        "tile": TILE, "args": vars(args)}, ckpt)
        else:
            sin_mejora += 1
        print(f"ép {ep:2d} | loss {fila['loss_train']:.3f}/{loss_va:.3f} | acc {fila['acc_train']:.3f}/{fila['acc_val']:.3f}"
              f" | F1 val {f1_va:.3f}{' *' if mejora else ''} | {fila['seg']:.0f}s", flush=True)
        pd.DataFrame(hist).to_csv(OUT / f"{args.tag}_historia.csv", index=False)
        if sin_mejora >= args.patience:
            print(f"Parada temprana: {args.patience} épocas sin mejorar", flush=True)
            break

    model.load_state_dict(torch.load(ckpt)["state_dict"])
    p_va, _ = predecir(model, x_va, y_va)
    t0 = time.perf_counter(); p_te, _ = predecir(model, x_te, y_te, batch=1 if len(y_te) < 50 else 256)
    ms = 1000 * (time.perf_counter() - t0) / len(y_te)
    preds = pd.concat([va.assign(p_mala=p_va), te.assign(p_mala=p_te)])[["path", "fruta", "calidad", "split", "y", "p_mala"]]
    preds.to_csv(OUT / f"{args.tag}_predicciones.csv", index=False)
    resumen = {
        "tag": args.tag, "params": n_params(model), "mejor_epoca": best_ep, "epocas": len(hist),
        "f1_val": f1_score(y_va.numpy(), p_va >= 0.5, average="macro"),
        "f1_test": f1_score(y_te.numpy(), p_te >= 0.5, average="macro"),
        "auc_test": roc_auc_score(y_te.numpy(), p_te), "ms_por_imagen_lote": ms,
        "min_total": sum(h["seg"] for h in hist) / 60, "args": vars(args),
    }
    (OUT / f"{args.tag}_resumen.json").write_text(json.dumps(resumen, indent=2))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in resumen.items() if k != "args"}, indent=1))


if __name__ == "__main__":
    main()
