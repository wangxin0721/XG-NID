# 候选缓存与配对分析执行（2026-10-23）

当前按用户确认继续现有计划：E3a-Z未完成；不重训E3a、不重跑E3a诊断；A/C/B及lambda=.05保持。新增analyze_candidate_fusion_cache.py是单文件标准库离线工具，无GPU/模型依赖，可与服务器训练并行。真实E3a候选缓存尚未同步本地，本机未执行完整缓存分析，不预记任何修复/退步数或扫描分布。

当前E3a全验证错误308个，其中至少一路H(z)正确但融合错122个(39.61%)，两路H(z)都错且融合错186个(60.39%)；另有94个两路都错样本被融合挽救。应同时观察分支变化、融合退步和最终Macro，而非单路准确率最大化。WebBased76个漏检中60个→Recon(78.95%)；71个误报中37个来自Recon(52.11%)。这些统计是本次诊断重点，不新增类别损失或类别专用门控。

## 工具回答的问题

1. 对仅Packet正确的WebBased成功/失败两组分别汇总实际权重、真实类别相对最高错误类别的概率差，以及距最近正确采样alpha的距离：样本数、均值、最小值、P10、中位数、P90、最大值；逐网格统计正确数。alpha指Flow权重，纯Packet为0。
2. 每个样本的alpha=0/1端点必须与H(z_packet)/H(z_flow)预测及真类间隔一致；alpha=.5与等权一致。检查网格预测与间隔符号、非连续正确网格段、最终权重合法性；不推断连续阈值、不增加网格、不生成逐样本oracle结果。
3. WebBased→Recon及Recon→WebBased均报告原E3/候选误判子集。分类为持续该方向错误、新引入、改为正确、变为其他错误；报告旧/新两路及融合的p(WebBased)-p(Recon)分布和预测类别计数。这是概率差，缓存未保存完整分类logits，不冒充原始logit差。
4. 根据sample_id配对，同时核对label/node_hash、有序身份摘要及完整样本集合。逐类及全体验证给出修复、退步、保持正确、持续错误，报告旧/新单路四分组转换。检查修复-退步与矩阵对角线净变化相符。
5. 从缓存重建原E3融合及候选融合/两路/等权矩阵，验证候选四分组统计；检查压缩文件SHA以及单独扫描文件与完整缓存相同子集逐条一致。源报告必须为完成状态且引用相同冻结参考/原诊断。工具不读取检查点，也不加载图。

输出report.json为聚合，可回传；paired_diagnostics.jsonl.gz含逐样本身份与诊断，仅留服务器outputs，不公开提交。正确率/单路变化是描述，不据修复计数单独判断Macro收益；多seed验收仍按原计划。

## 服务器命令

```bash
(
  set -e
  cd /home/nzxn/下载/WX/load/XG-NID
  git fetch origin codex/r0-edge-preparation
  cache_rev=$(git rev-parse FETCH_HEAD)
  cache_tool=$(mktemp /tmp/xgnid_candidate_cache.XXXXXX.py)
  git show "$cache_rev:scripts/analyze_candidate_fusion_cache.py" > "$cache_tool"
  nice -n 10 /home/nzxn/miniconda3/envs/wx/bin/python -B -u "$cache_tool" \
    --original-report outputs/e3_gate_diagnostic_b512_s42_v1/report.json \
    --candidate-report outputs/e3_gate_candidate_E3a_b512_s42_v1/report.json \
    --output outputs/e3_gate_candidate_cache_E3a_b512_s42_v1
)
```

目录已存在则拒绝覆盖；检查已有status/report，不自动追加版本重跑。请使用服务器原report.json，不复制聊天文本替代源报告（此前粘贴文本少了一个逗号，服务器原件状态未由此判定损坏）。

完成后：

```bash
cat outputs/e3_gate_candidate_cache_E3a_b512_s42_v1/status.json
cat outputs/e3_gate_candidate_cache_E3a_b512_s42_v1/report.json
```

E3a-Z完成并取得自己的候选诊断缓存后，同工具替换candidate-report与output即可；原E3参考保持。当前不排新训练。

本地7项标准库检查通过，覆盖非单调扫描/最近采样点、端点及等权不一致拒绝、错误网格段与权重拒绝、配对修复退步、双向概率差、空组/非有限输入、Python3.9语法及无ML导入。真实缓存端点与配对统计须待服务器运行验收，以上检查不冒充真实分析已完成。
