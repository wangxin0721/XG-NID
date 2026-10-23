# 候选诊断交付与A/C/B设置（2026-10-23）

已实现scripts/diagnose_gate_candidate.py，可在E3a-Z继续训练时独立诊断已完成的E3a。脚本只读取训练初态所需图和完整验证集；不构造测试loader、不训练、不修改检查点、不执行全局可学习权重训练。A/C/B设计固定在configs/fusion_acb_study_20261023.json，标记为设计登记，训练入口尚未实现，不增加本轮训练任务。

辅助损失固定为融合CE + 0.05/2 × [CE(H(z_flow),y)+CE(H(z_packet),y)]。直接监督H(z)，原z及其熵/间隔仍叫表示统计。A固定等权；C从s=0开始，训练一个全局sigmoid标量；B最后一层零初始化，初始等权且保持可训练。三者共用编码器、投影、共享非线性头初态与训练顺序；均取消最终代理混合。该组三项只能隔离新增监督条件下的权重机制，不能解释相对旧实验的全部改动贡献。

## 诊断输出

- 完整验证矩阵复现后才记completed；原模型/检查点保持不变；明确记录候选variant与独立文件哈希。
- report.json：全体/逐类四组样本分母、占比、正确/错误及保留/挽救率；空组比例null；逐类Precision/Recall/F1、误报/漏报及双向混淆；单路H(z)及融合的Macro；NLL/Brier和10箱校准统计。
- validation_cache.jsonl.gz：各样本实际最终权重、分支表示、H(z)与实际融合概率、top1-top2间隔、真类有符号间隔、中心化logits尺度、21点真实共享头扫描。原z不标为分类概率。
- webbased_packet_only_scan.jsonl.gz：仅Packet的H(z)正确的全部WebBased样本子集，包括被保留和被破坏的样本，而非只导出失败案例。
- alpha定义为Flow权重，Packet=1-alpha。扫描使用实际H(alpha*zf+(1-alpha)*zp)，缓存全部网格预测和真类间隔，可出现不相连的正确网格段；不宣称网格段之间全部连续正确。实际门控权重及其融合间隔另外保存。
- 逐样本扫描使用标签仅作事后解释，不输出oracle预测文件/准确率，不用于选逐样本推断权重。报告中的最佳全局网格alpha也明确是检查点干预，不等同从头训练的C。
- 参数量与batch16原模型CUDA前向时间单独记录，不含额外诊断头扫描；共享GPU时不能宣称独占速度。压缩逐样本缓存留在服务器outputs，不提交公开仓库。

旧E3的WebBased仅Flow正确组失败1/246=0.41%，仅Packet正确组失败56/196=28.57%；后者占全部81个错误69.14%。这不是“偏爱Flow”的因果证明。新候选须重算。对WebBased真类的四组表只描述召回，F1必须结合来自其他类别的误报。

## 服务器执行E3a

以下在独立临时目录提取工具，fetch不覆盖服务器工作树。通过元数据检查后只提交诊断任务，不串联新训练。默认物理GPU0显式共享，可统一换成1；无需暂停E3a-Z。输出已存在时停止，避免重复提交。

```bash
(
  set -e
  cd /home/nzxn/下载/WX/load/XG-NID
  git fetch origin codex/r0-edge-preparation
  candidate_rev=$(git rev-parse FETCH_HEAD)
  candidate_tools=$(mktemp -d /tmp/xgnid_candidate_diag.XXXXXX)
  for name in e3_gate_study.py diagnose_e3_gate_server.py run_e3_gate_ablation_server.py diagnose_gate_candidate.py; do
    git show "$candidate_rev:scripts/$name" > "$candidate_tools/$name"
  done
  git show "$candidate_rev:configs/e3_gate_study.json" > "$candidate_tools/e3_gate_study.json"

  candidate_out=outputs/e3_gate_candidate_E3a_b512_s42_v1
  if [ -e "$candidate_out" ] || [ -e "$candidate_out.log" ] || [ -e "$candidate_out.pid" ]; then
    echo "已有诊断产物，请先查看status和日志；不重复启动。"
    exit 1
  fi
  /home/nzxn/miniconda3/envs/wx/bin/python -B -u "$candidate_tools/diagnose_gate_candidate.py" \
    --repo . --config "$candidate_tools/e3_gate_study.json" \
    --variant E3a --mode check \
    --screen-report outputs/e3_gate_screen_E3a_b512_s42_v2/summary.json \
    --diagnostic-report outputs/e3_gate_diagnostic_b512_s42_v1/report.json

  nohup /home/nzxn/miniconda3/envs/wx/bin/python -B -u "$candidate_tools/diagnose_gate_candidate.py" \
    --repo . --config "$candidate_tools/e3_gate_study.json" \
    --variant E3a --mode diagnose \
    --screen-report outputs/e3_gate_screen_E3a_b512_s42_v2/summary.json \
    --diagnostic-report outputs/e3_gate_diagnostic_b512_s42_v1/report.json \
    --physical-gpu 0 --allow-shared-gpu --output "$candidate_out" \
    > "$candidate_out.log" 2>&1 < /dev/null &
  echo "$!" > "$candidate_out.pid"
  echo "诊断请求已提交，请核对日志和最终status。"
)
tail -n 20 -F outputs/e3_gate_candidate_E3a_b512_s42_v1.log
```

完成后查看outputs/e3_gate_candidate_E3a_b512_s42_v1/status.json与report.json。无需回传逐样本缓存即可先分析汇总。

E3a-Z完成后，同样命令只替换variant为E3a_equal_init、screen-report为outputs/e3_gate_screen_E3aZ_b512_s42_v1/summary.json、candidate_out为outputs/e3_gate_candidate_E3aZ_b512_s42_v1。入口拒绝未完成的screen，不能用旧E3报告替代候选。

## 验收和当前验证范围

seed42按验证Macro及逐类/机制结果筛选；B超过A/C只能提名确认，小幅领先不作为增加模块的理由。配置冻结后比较seed43/44，与已披露的探索seed42逐seed配对报告差值及平均/样本标准差。B接近C或收益不稳定时暂停扩展动态结构，不要求每个seed获胜；同时记录E0/E2及候选成本。测试在配置确定后统一评价。

本地7项标准库检查通过：Python3.9语法/无ML导入、非单调扫描、真类间隔、空组分母、误报与F1、拒绝未完成/错variant/测试参与、明确损失及无时间预算。真实Linux GPU诊断尚未执行；需服务器运行后复现保存验证矩阵才验收。不把标准库检查当作GPU验证通过。
