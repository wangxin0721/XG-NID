# R0服务器执行说明（待真实执行）

本地开发→独立Git提交/GitHub分支→服务器检查。所有以下命令均未在服务器运行。不要中断已运行E1；确认GPU空闲后才执行R0 CUDA冒烟。R0不能直接使用E0的分组split。环境名尚未回传，以下用必填变量，不能假装已确认是nids。

## 1. 先确认E1与E0交付

在已有服务器终端、仓库根目录运行，无需拉代码即可检查：

```bash
date -Is
pwd
pgrep -af '[p]ython.*train_grouped.py' || true
find outputs -maxdepth 3 -type f \( -name '*.pid' -o -name status.json -o -name summary.json \) -path '*E1*' -print
```

从进程`--output`参数或找到的文件确认**实际**E1目录；不要凭示例名字认定它已启动。进程不存在、status是running时可能异常退出；有completed也要核对对应配置/日志/exitcode。PID可能复用，核对ps命令内容和启动时间，不能只kill -0。

同步下述代码后（Python若尚不可用先执行第2节环境激活）：

```bash
export E1_RUN='替换为刚确认的E1输出目录'
python scripts/collect_experiment_status.py --run-dir "$E1_RUN" \
  --pid-file "$E1_RUN.pid" --log-file "$E1_RUN.log"
python scripts/collect_experiment_status.py \
  --run-dir outputs/grouped_full_E0_s42_v1 \
  --pid-file outputs/grouped_full_E0_s42_v1.pid \
  --log-file outputs/grouped_full_E0_s42_v1.log
```

上例.pid/.log是检查候选；若真实启动命令使用其他名字，改成真实路径；缺少PID文件不等于没运行。E0目录按既有交接命令填写，若实际输出不同，以真实目录为准。需回传采集JSON、进程行、真实环境名；数据/权重留服务器。完整E0需summary/status、E0/config/epochs/best/last/metrics/predictions、外部日志与exitcode（若原启动器保存）齐全，不能只回一行分数。

## 2. 拉取独立交付并激活已有环境

```bash
git status --short
git branch --show-current
git fetch origin codex/r0-edge-preparation
```

先保留服务器已有改动；若工作树干净且没有正在写源码的操作，再切换分支。正在运行的E1不需要中断，但不能修改其冻结模型源码或结果目录；此分支只新增R0工具/文档。

```bash
git switch --track origin/codex/r0-edge-preparation
# 若这个本地分支已存在，改用 git switch codex/r0-edge-preparation，再 git pull --ff-only
git rev-parse HEAD
source "$(conda info --base)/etc/profile.d/conda.sh"
export XGNID_ENV='替换为现有训练conda环境名'
conda activate "$XGNID_ENV"
python -c 'import sys,torch,torch_geometric,sklearn; print(sys.executable); print(torch.__version__,torch_geometric.__version__,sklearn.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA")'
nvidia-smi
```

不要运行旧bootstrap，不重装CUDA/torch。记录实际环境与官方requirements差异。Git HEAD应与本次聊天给出的交付SHA相同；服务器如果尚无该GitHub分支，先完成本地推送，不能用待推送代码当已同步。

## 3. 恢复当前源train/test清单（CPU，只读图）

已知服务器旧图候选目录是`data/processed/CICIoT2023_processed/processed`，先核对实际存在。保留源CSV、图、历史审计SQLite及split。不要运行原始预处理notebook：它包含覆盖CSV/删除中间文件的操作。

```bash
export R0_DATA='data/processed/CICIoT2023_processed/processed'
test -d "$R0_DATA"
python - <<'PY'
import os,re,collections
from pathlib import Path
root=Path(os.environ['R0_DATA'])
counts=collections.Counter()
examples={}
for p in root.rglob('*.pt'):
    m=re.fullmatch(r'data_(test_)?(\d+)\.pt',p.name)
    part=('test' if m[1] else 'train') if m else 'other'
    counts[part]+=1
    examples.setdefault(part,str(p))
print(dict(counts)); print(examples)
PY
```

若没有官方`data_N.pt`/`data_test_N.pt`命名，必须从可信构图记录/源CSV映射得到locations.jsonl，例如每行`{"path":"shard.pt","container_offset":0,"partition":"train"}`。这是**成员身份映射**，不能拿旧按类随机切分的split充当来源。若无法确认源train/test对应关系，本步骤停在待补证据，不运行full，不通过重新选成员补齐数量。

准备一个真实来源说明文件；内容至少包含源图路径、来源/下载记录、源train/test CSV实际路径与SHA256、文件名或locations如何对应源CSV、缺失的作者样本/capture身份，以及已知重叠限制。可在JSON中如实写unknown；不能写“严格原始划分已恢复”。以下命令要求先设置这个文件的真实路径：

```bash
export R0_EVIDENCE='替换为真实来源说明文件路径'
test -s "$R0_EVIDENCE"
export R0_MANIFEST="outputs/r0_manifest_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p outputs
python -u scripts/r0_data.py --trusted-data "$R0_DATA" \
  --membership-evidence "$R0_EVIDENCE" --output "$R0_MANIFEST" \
  > "$R0_MANIFEST.log" 2>&1
R0_RC=$?
printf '%s\n' "$R0_RC" > "$R0_MANIFEST.exitcode"
tail -n 40 "$R0_MANIFEST.log"
cat "$R0_MANIFEST/status.json"
cat "$R0_MANIFEST/manifest.json"
```

非官方命名时在上述命令增加`--locations /实际位置/locations.jsonl`。目录扫描会完整读取约185557图，属CPU/I/O检查，会与E1争用磁盘/CPU；训练繁忙时安排合适时段。查看manifest里的逐类数量、table4_counts_match、Edge/NoEdge overlap、冲突标签及来源说明，不擅自去重或改测试成员。

表4数量一致只是必要检查；如果源成员可追溯但现有样本有重复，R0如实保留并披露，不能把结果作为无泄漏泛化证据。当前交付始终标注protocol_reimplementation。冲突标签、异常特征/拓扑或来源成员不能解释时先回传排查，不启动full。图变更会在实际加载时被哈希检查拒绝。

## 4. GPU空闲后只启动真实R0冒烟

```bash
export R0_SMOKE="outputs/r0_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
nohup bash -c '
python -u scripts/run_r0_edge.py --mode smoke --manifest-dir "$1" --device cuda:0 --workers 0 --output "$2" &
child=$!
printf "%s\n" "$child" > "$2.pid"
wait "$child"
rc=$?
printf "%s\n" "$rc" > "$2.exitcode"
exit "$rc"
' _ "$R0_MANIFEST" "$R0_SMOKE" > "$R0_SMOKE.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$R0_SMOKE.launcher.pid"
```

同一终端保留R0_MANIFEST/R0_SMOKE值，另一个终端需重新赋为**已有**真实目录。检查：

```bash
cat "$R0_SMOKE.pid"
ps -p "$(cat "$R0_SMOKE.pid")" -o pid,ppid,lstart,etime,args
tail -n 60 "$R0_SMOKE.log"
test ! -f "$R0_SMOKE.exitcode" || cat "$R0_SMOKE.exitcode"
test ! -f "$R0_SMOKE/status.json" || cat "$R0_SMOKE/status.json"
test ! -f "$R0_SMOKE/summary.json" || cat "$R0_SMOKE/summary.json"
```

smoke从源train按每类最多4张取32图，初始化/六层关系属性hook、边投影梯度和隔离扰动、Adam一步、2轮小训练、last重载，**不加载测试图计算指标**。失败需查看error.log/外部log；smoke不得复用已存在输出目录，修正后新建目录。不把CPU测试报告传给full。

验收必须同时满足：status completed/exit0、外部exitcode0、summary mode=smoke、runtime.device_type=cuda、optimizer_step_passed=true、epochs_completed=2、六个GAT属性非空、有限loss/gradient、至少一条可辨识属性效应超过抖动容差、非零边投影梯度、last.pt存在且SHA匹配。单入边Contain零效应可解释，不能删边或造属性来强行让所有关系有效。

**先回传manifest.json、membership_evidence.txt、smoke summary/status/log/exitcode，并人工核查。以下正式命令只做交付，不与冒烟串联自动执行。**

## 5. 冒烟真实通过后才可手动启动正式训练

启动条件：E1状态已真实确认且不抢占现有任务；来源成员和特征顺序限制已说明；逐类表4数量一致；GPU可用；同manifest/config/source/runtime的CUDA smoke通过且人工审阅；没有按测试结果挑数据/参数。当前尚未满足服务器部分。

```bash
export R0_FULL="outputs/r0_full_s42_$(date -u +%Y%m%dT%H%M%SZ)"
nohup bash -c '
python -u scripts/run_r0_edge.py --mode full --manifest-dir "$1" --smoke-report "$2/summary.json" --device cuda:0 --workers 0 --output "$3" &
child=$!
printf "%s\n" "$child" > "$3.pid"
wait "$child"
rc=$?
printf "%s\n" "$rc" > "$3.exitcode"
exit "$rc"
' _ "$R0_MANIFEST" "$R0_SMOKE" "$R0_FULL" > "$R0_FULL.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$R0_FULL.launcher.pid"
```

固定30轮、lr0.01、batch64、测试batch1、Adam无weight decay/调度器/早停，不划val，不搜索测试指标，不额外开启E2/E3或M1。每100batch/每轮写日志；每轮记录训练集Accuracy，保存last，不存在按测试分数选best。没有自动恢复入口，失败保留目录和日志，先查原因，不自动重跑挑最佳。

监控及最终读取：

```bash
ps -p "$(cat "$R0_FULL.pid")" -o pid,ppid,lstart,etime,args
tail -n 60 "$R0_FULL.log"
test ! -f "$R0_FULL/epochs.jsonl" || tail -n 3 "$R0_FULL/epochs.jsonl"
test ! -f "$R0_FULL/status.json" || cat "$R0_FULL/status.json"
test ! -f "$R0_FULL.exitcode" || cat "$R0_FULL.exitcode"
# 确认完成后：
cat "$R0_FULL/summary.json"
cat "$R0_FULL/metrics.json"
head -n 4 "$R0_FULL/predictions.csv"
sha256sum "$R0_FULL/last.pt" "$R0_FULL/predictions.csv"
```

需summary completed/exit0、epochs_completed30、selection last_epoch、测试samples25557及逐类support正确、预测25557条（另有表头）、log/exitcode一致，才能认定一次R0训练交付完成；训练完成仍不能直接宣称严格复现论文分数。权重、大型预测、CSV和完整数据清单保留服务器，GitHub仅交付源码与小型审查报告。

若R0来源/冒烟尚未就绪，按既有冻结入口继续已准备好的E2/E3，不改预算和loss，也不为R0中断正在运行的任务。本说明没有启动E2/E3。
