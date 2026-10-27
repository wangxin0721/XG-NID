# B-H batch1024：单次探索运行

用户请求：另备一版训练 batch1024，之后手动运行。代码交付时尚未执行 GPU 探测或训练。

仅训练 batch 由512改为1024；验证16、初始化16、seed42、Adam lr0.001、weight decay0、λ0.05、最多100轮、patience12和严格验证 Macro-F1 选模规则保留。
物理 GPU1，允许共享。每轮107621个样本，106次优化更新；batch512为211次。
使用同一初态证书和采样器的逐轮随机排列，批次分组及优化轨迹会改变。
模型、H统计输入、无detach的梯度路径及辅助监督均复用B-H实现。

入口：`scripts/run_fusion_bh_b1024_server.py`；配置：`configs/fusion_bh_b1024_study_20261027.json`。
独立目录：`outputs/fusion_bh_b1024_s42_v1`，日志同名加`.log`，拒绝覆盖已有产物。
流程：三批探测 → 新进程从干净初态正式训练 → 最佳检查点验证 → 配对诊断及分析。
batch512入口、配置及结果保持原有版本，不会自动启动1024。

服务器执行（外层终端可以是sh，以下显式使用bash）：

```bash
bash <<'BASH'
set -euo pipefail
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
bundle=$(mktemp -d /tmp/xgnid_bh1024.XXXXXX)
git -c core.autocrlf=false archive FETCH_HEAD scripts configs | tar -x -C "$bundle"
bash "$bundle/scripts/launch_fusion_bh_b1024.sh" 1
BASH
```

交付消息提供固定提交号，可替代FETCH_HEAD。
查看日志：`tail -n 30 -F outputs/fusion_bh_b1024_s42_v1.log`。
完成后回传根目录`summary.json`和`analysis.json`。

主要将此结果与B-H batch512比较，判断batch改变的影响。报告仍列出B/C/E2/E0的指标与配对统计，但它们训练batch为512，不能把差异全部解释为门控统计输入的收益。
报告不会据此自动建议或启动更多seed。单次结果不代表稳定提升。

本地验证：原B-H五项及新增四项标准库测试通过；真实本地证据绑定、Python语法和Bash语法检查通过。
未导入torch、未反序列化权重；GPU显存与梯度检查由服务器三批探测完成。
