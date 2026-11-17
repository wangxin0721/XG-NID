# 最终模型的逐维动态门控消融

在 seed42、43、44 的已锁定 B 上做配对比较。每个种子内，两组共用冻结的 Flow/Packet 编码器、报文节点读出、报文转移读出和两路 32 维投影。seed44 沿用验证选中的第 0 轮报文转移读出，其残差为零。

| 组别 | 门控使用的输入 | 融合方式 |
| --- | --- | --- |
| Static | 该主干在全部训练流上的 69 维输入均值 | 所有流共用一组可学习的 32 维权重 |
| Dynamic | 当前流自己的 69 维输入 | 每条流产生自己的 32 维权重 |

两组复制同一份 C 的门控网络和共享分类头，将门控末层归零，以相同的 0.5 逐维权重开始训练。每组有 2,984 个可训练参数。使用 Adam、学习率 0.001、训练批量 512、相同 seed42 采样顺序和八类 NLL。每组 30 轮、6,330 次更新。第 0 轮到第 30 轮按全验证集 Macro-F1 选模，同分取较早轮次。原 B 只作为参考成绩。

## 在已有 GPU 环境启动

在完成独立主干测试的服务器上，激活该实验使用的 Python 3.9.23、Torch 2.3.1+cu118、PyG 2.5.3 环境。选择本次运行的可用 GPU。将 `XG-NID_Final_D32_Gate_Ablation.patch` 放在主目录。下面从现有仓库读取已归档的权重、划分和原图，用独立工作树运行新代码。

```bash
cd '/home/nzxn/下载/WX/load/XG-NID'
xgnid_repo="$PWD"
xgnid_code="$(dirname "$PWD")/XG-NID-final-d32-gate"
git worktree add --detach "$xgnid_code" 3adaefe728098a67075bac14832b325573c1ed84 &&
git -C "$xgnid_code" apply "$HOME/XG-NID_Final_D32_Gate_Ablation.patch" &&

nohup bash -c '
  set -eu
  export PYTHONPATH="$2/src${PYTHONPATH:+:$PYTHONPATH}"
  for seed in 42 43 44; do
    python -u "$2/scripts/run_final_d32_gate_ablation.py" --repo "$1" --seed "$seed"
  done
  python -u "$2/scripts/run_final_d32_gate_ablation.py" --repo "$1" --test
' _ "$xgnid_repo" "$xgnid_code" > outputs/final_d32_gate_ablation_v1.log 2>&1 < /dev/null &
```

运行中查看 `outputs/final_d32_gate_ablation_v1.log`，各组进度在 `outputs/final_d32_gate_ablation_v1/seed*/progress.json`。如需显式指定原始图根目录，在每条 Python 命令末尾加 `--data-root /实际图目录`。若某次运行中断，可以使用新的 `--output outputs/final_d32_gate_ablation_rerun` 重新执行配对流程，原目录可留作排查。

训练入口读取 107,621 张训练图及 13,446 张验证图。测试入口确认三份验证选模记录和权重后，逐条验证回放并在固定的 13,448 张测试图上打分。主结果位于 `outputs/final_d32_gate_ablation_v1/test/report.md`。报告以 Dynamic 对 Static 的配对差为主，同时给出逐类 F1、两类双向错分和逐样本预测。
