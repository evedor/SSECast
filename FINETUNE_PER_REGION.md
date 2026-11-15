# 多区域预训练 + 单区域微调（最后合并成一个 ckpt）

## 它做什么

```
config/multiregion_l2.yaml (train.py)          →  共享主干 + 4 个区域 adapter/head（loss=L2）
finetune_per_region.py (逐区域)                →  固定主干，只训 region_adapters.<r> + region_heads.<r>
                                                  其它三个区域的头 / region_embed / head1/head2/head_out 全部冻结
最后 merge_and_save()                          →  <out_dir>/<tag>.ckpt  == 一个模型（共享主干 + 每区自己微调过的头）
```

* 微调时区域之间**没有参数共享**，所以"分开微调、最后仍是一个模型"是自洽的：主干逐位不变，
  四个区域模块各自落在自己的 key 上，合并不会冲突。脚本里对此有硬断言：
  `[merge] 改动的张量 N 个，全部属于 [...] 的区域模块（其余 489 个与预训练逐位相同）`。
* 产物结构与 `train.py` 完全一致（`model_state` / `optimizer_state_dict` / `meta`），
  `ipynb/predict_cascadia.ipynb` 的 `Load_model` 不用改，只要把 `best_checkpoint_path` 指到新 ckpt。

## 命令

```bash
cd experiments_1104

# ① 多区域预训练（L2）
python train.py --yaml_config ./config/multiregion_l2.yaml --config multiregion

# ② 逐区微调 + 合并（四区一次做完，产出单个 ckpt）
python finetune_per_region.py \
  --base_yaml ./config/multiregion_l2.yaml --base_config multiregion \
  --ft_config l2 --tag all_l2 \
  --ft_regions cascadia,alaska,nankai,newzealand \
  --pretrained_ckpt_path ./Results/multiregion_l2/00/training_checkpoints/best_model.ckpt \
  --out_dir ./Results/multiregion_l2/finetune_per_region \
  --epochs 20 --patience 5

# 只微调一个区域（先验证方向）
python finetune_per_region.py --ft_config l2 --tag cascadia_l2 --ft_regions cascadia --epochs 20 --patience 5

# 全流程串起来（预训练 → 微调 → 评测）
bash /tmp/run_l2_pipeline.sh
```

## 关键开关（config/finetune_per_region.yaml）

| 键 | 含义 |
|---|---|
| `pretrained_ckpt_path` | 微调起点；必须与 `base_yaml` 同架构 |
| `ft_regions` | 要微调哪些区域（顺序即执行顺序） |
| `keep_adapters` | `true`=heads+adapters；`false`=只训 head（≈线性探针） |
| `loss_mode` | 微调损失口径；`l2` 修幅度压缩，`l1` 与 L1 预训练一致 |
| `lr` / `max_epochs` / `early_stopping_patience` | 每区域独立跑这套预算 |

**不要**在微调 yaml 里改 `regions` / `patch_size` / `token_*` / `pos_embed_mode`：
架构必须与预训练 ckpt 一致，改了脚本会直接抛错（它用 `strict` 加载并校验 missing/unexpected）。

## 已知结论（2026-11-08 实测，test split、starts=1..364、notebook 同口径）

| 模型 | lead1 RMSE | lead1 ACC | lead1 Δ幅度比 | Δ相关 |
|---|---|---|---|---|
| 旧单区域 finetune（`Results/cascadia/11`） | **0.0574** | **0.9969** | **0.86** | **0.82** |
| 多区域 l1 预训练（`multiregion/00/best_model`） | 0.0868 | 0.9923 | 0.32 | 0.41 |
| ↑ 再用 **l2 只微调 head+adapter**（cascadia） | **0.0730** | 0.9949 | 0.57 | 0.66 |
| 多区域 L2 预训练 → 逐区 l2 微调 | 本轮产物 | | | |

结论：幅度压缩的主因是 L1（中位数回归），只换 L2 微调头就能把 lead1 误差降 16%、Δ 幅度从 32% 提到 57%。
预训练也用 L2 是下一步（本轮跑的就是它）。
