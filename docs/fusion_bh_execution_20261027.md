# B-H 执行交付（2026-10-27）

实现已完成；本地仅运行标准库检查，服务器三批探测与正式训练尚未运行。
原始方案见 `下一轮B-H实验计划_20261027.md`。

## 固定修改

独立候选 `B_Hstats` 复用旧 B 的完整初态、训练起点 RNG、损失函数与数据顺序。
只把 21 维门控输入的第 16–19 列（从零计数）换成 `softmax(H(z))` 的熵补量与前两概率差。
16 维表示和表示余弦相似度不变，门控仍是 21→16→2，最终预测仍为 `H(wf*zf+wp*zp)`。
不增加 detach，不引入标签、代理混合、独立分类头或类别加权。
新增 H→统计→门控梯度路径属于本次改变；概率统计不等于经过校准的置信度。

seed42、train512/eval16/init16、Adam lr0.001、weight decay0、λ0.05、最多100轮、patience12、严格验证 Macro-F1 选模。
不设 GPU 小时限制；不自动安排其他 seed 或候选。使用物理 GPU1，允许共享。

为了原样复用旧 B 损失函数，两路辅助监督的 H(z) 会重新计算。
B-H 模型前向调用 H 三次，训练损失合计五次；旧 B 分别为一次、三次。
参数量不变，但计算量增加；共享 GPU 的耗时不能直接作为独占速度对照。

## 自动执行内容

1. 标准库检查 A/C/B、强基线、数据划分、冻结源码及初始化证书；检查点只作字节哈希。
   只依赖 Z 的已完成预测缓存，不读取本地未同步完整的 Z 权重和训练日志。
2. 三个训练 batch 的一次性探测：初始等权、B/B-H 初始输出及四项损失一致、切回旧输入恢复 B、标签置换不影响门控、四维替换、辅助监督梯度、门控末层与前层更新、新 H 梯度路径。
   审计前向后恢复所有参数、缓冲区和 RNG；审计本身零优化步。
3. 新进程从干净初态训练 B-H，探测更新不进入正式训练。逐轮核对已存在的旧 B 74 轮数据顺序；其后继续同一冻结采样器。
4. 最佳检查点完整验证，导出私有逐样本预测；自动复算指标、B/C 配对修复和退步、Z 挽救子集保留率、64 个 WebBased 样本恢复数量及全部新增错误。
5. 按完整精度阈值输出停止/确认建议；不执行测试集，不自动扩训。

输出根目录：`outputs/fusion_bh_b512_s42_v1`。
结束后查看 `status.json`、`summary.json`、`analysis.json`、`decision.md`。
候选的 `B_Hstats/metrics.json` 和 `diagnostics.json` 保存选模与逐类结果。
`validation_predictions.jsonl.gz`、权重和数据保持私有，不推送公开仓库。

## 服务器命令

从本次交付提交导出独立脚本包；不覆盖服务器旧 checkout：

```bash
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
bundle=$(mktemp -d /tmp/xgnid_bh.XXXXXX)
git -c core.autocrlf=false archive FETCH_HEAD scripts configs | tar -x -C "$bundle"
bash "$bundle/scripts/launch_fusion_bh.sh" 1
tail -n 30 -F outputs/fusion_bh_b512_s42_v1.log
```

为严格复现，优先使用交付消息给出的固定提交号替代 FETCH_HEAD。
启动器拒绝覆盖已有输出和日志，探测失败即停止队列；不要删除旧结果后盲目重试。

## 本地验证

5 项标准库单元测试通过；新增 Python 语法检查、Bash 语法检查、真实本地证据哈希核验通过。
另用旧 B 缓存作为明确标记的模拟候选验证报告流程：B→自身零差异、64例恢复0、Z原挽救保留220，均符合预期。
这不是 B-H 训练结果。未导入 torch、未反序列化权重、未执行真实前向或反向。
完整记录见 `fusion_bh_local_validation_20261027.json`。
