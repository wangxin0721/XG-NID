# R0-A：对齐官方数字索引与延迟初始化

> 后续状态：v1由用户主动停止，未形成30轮结果；全量重生图已确认185557张输入/标签与旧图一致。用户要求继续后，相对路径探测返回last.pt不存在，当前使用[r0a_restart_20261019.md](r0a_restart_20261019.md)中的独立v2启动器再次核对绝对路径，无检查点才从第1轮重启。不要重复运行下文旧v1启动命令。

当前R0完成30轮，测试Macro-F1为92.8211%，与论文图示约97%仍有差距；不得将其称为性能复现成功。已有服务器检查未发现确定的模型封装、受控训练单步、指标计算或所检查输入数值错误。本次继续原论文复现，先处理两处已确认的执行差异，不全量重建数据。

## 固定比较范围

独立入口`scripts/run_r0_aligned_server.py`通过原`outputs/r0_full_launch.json`定位完成训练的旧worktree、清单和产物，核验原源码、官方Model/Training锁、配置、清单及原checkpoint。只在Linux服务器加载数据和模型。

| 项目 | 原R0 | 独立R0-A |
| --- | --- | --- |
| 基础索引 | 文件名字典序 | 官方`data_0.pt`至`data_159999.pt`数字顺序；要求编号连续且成员唯一 |
| lazy初始化 | 正式训练前有eval预热 | 无额外前向，首次训练batch完成初始化 |
| 模型/训练 | 官方Edge包装及等价循环 | 直接官方`HeteroGNN_Edge`与锁定`train_with_edge_Att`/`test_edge`函数 |
| 数据成员与输入 | 原冻结图 | 相同清单、图、特征、标签，逐图沿用冻结指纹核验 |
| 控制参数 | seed42，Adam0.01，64/1，30轮，最后一轮 | 相同，无调度、早停或测试选模 |

官方训练函数通过AST提取，保留实际训练及训练集评价语句；在optimizer.step之后和epoch结束处分别附加观察、进度与checkpoint记录。保存实际执行源码便于核查。不在训练前探测模型；首次真实前向记录6个GAT的边属性输入，首次真实更新核对有限梯度、参数确有更新及边投影非零梯度；四个BN计数必须与训练步数一致。官方按同一shuffle训练loader评价训练准确率的调用保留，相关随机消费不删除。

原R0的受控单步已证明两条路径在共享初始状态下相同，这里直接调用官方函数进一步减少独立循环差异。本次同时对齐顺序和初始化，属于联合对照，单次结果不能分离两者各自影响，也不能估计种子波动或证明历史作者CUDA环境相同。原作者训练种子和样本身份仍未知，R0-A仍标为protocol_reimplementation；不许因结果较高就删除原R0或宣称已完全复现。

## 现在只执行CUDA冒烟

在服务器原仓库wx环境执行以下命令，不pull、不切换旧worktree。限定256张训练图，每类32张，按数字顺序固定选取；2轮，batch64，不读取测试图。脚本拒绝覆盖既有输出，并检查其他已知R0/分组训练进程；该检查不等于整台服务器GPU资源全局空闲保证。

```bash
. /home/nzxn/miniconda3/etc/profile.d/conda.sh && conda activate wx
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
git show origin/codex/r0-edge-preparation:scripts/run_r0_aligned_server.py > /tmp/xgnid_r0_aligned.py
CUDA_VISIBLE_DEVICES=0 python -B -u /tmp/xgnid_r0_aligned.py --repo . --mode smoke --device cuda:0 --output outputs/r0_aligned_smoke_s42_v1
```

回传终端最终JSON。通过应为status=completed、exit_code=0、epochs_completed=2、optimizer_step_passed=true、test=null；模型参数应为431704，6个GAT接收到边属性，至少一项边投影梯度非零。原checkpoint在运行前后核验不变。失败不自动重试、不重建数据、不调整参数；保留error.log。

## 冒烟通过后才执行一次正式对照

新入口不自动启动正式训练。正式模式强制核对新R0-A冒烟的代码、数据、配置、运行环境绑定及其checkpoint SHA；旧R0冒烟不能替代。2026-10-19已收到新冒烟通过回传，详见末节。下面为前台运行命令，需保持终端连接；正式训练尚未收到启动证据。

```bash
CUDA_VISIBLE_DEVICES=0 python -B -u /tmp/xgnid_r0_aligned.py --repo . --mode full --device cuda:0 --smoke-report outputs/r0_aligned_smoke_s42_v1/summary.json --output outputs/r0_aligned_full_s42_v1
```

需要后台运行时，使用下面命令**代替**前台命令，不能两者都运行。先在原仓库wx环境执行；复用已通过冒烟的`/tmp/xgnid_r0_aligned.py`，无需更新脚本或重跑冒烟。外层先用noclobber独占创建日志，拒绝覆盖已有输出/日志；脚本自身仍检查已知训练进程和匹配冒烟。nohup返回只表示提交启动，不表示正式训练已正常开始。

```bash
(
  set -C
  if [ -e outputs/r0_aligned_full_s42_v1 ]; then
    echo 'R0-A output already exists; stop.'
    exit 1
  fi
  : > outputs/r0_aligned_full_s42_v1.log || exit 1
  nohup sh -c '
    printf "%s\n" "$$" > outputs/r0_aligned_full_s42_v1.supervisor.pid
    CUDA_VISIBLE_DEVICES=0 python -B -u /tmp/xgnid_r0_aligned.py --repo . --mode full --device cuda:0 --smoke-report outputs/r0_aligned_smoke_s42_v1/summary.json --output outputs/r0_aligned_full_s42_v1 &
    train_pid=$!
    printf "%s\n" "$train_pid" > outputs/r0_aligned_full_s42_v1.pid
    wait "$train_pid"
    rc=$?
    printf "%s\n" "$rc" > outputs/r0_aligned_full_s42_v1.exitcode
    exit "$rc"
  ' >> outputs/r0_aligned_full_s42_v1.log 2>&1 < /dev/null &
)
```

连续显示日志（Ctrl+C仅停止查看，不终止后台训练）：

```bash
tail -n 30 -f outputs/r0_aligned_full_s42_v1.log
```

在另一终端可核对训练PID和最终退出码；exitcode在进程结束后才生成，缺失不等于失败：

```bash
ps -p "$(cat outputs/r0_aligned_full_s42_v1.pid)" -o pid,ppid,etime,args
cat outputs/r0_aligned_full_s42_v1/status.json
cat outputs/r0_aligned_full_s42_v1.exitcode
```

正式训练固定30轮，每轮记录训练loss/accuracy、训练样本身份顺序摘要、优化步数和用时；不额外增加训练F1全量扫描。保存最后一轮权重并加载核对，然后对同一现有测试成员评价一次，保存metrics、逐样本身份/标签/预测和CM。原R0和E0～E3不被修改，既有CSV与图全程只读。测试数字重排本身不会改变固定模型的汇总F1。

查看进度和结果（在另一终端执行）：

```bash
tail -n 3 outputs/r0_aligned_full_s42_v1/epochs.jsonl
cat outputs/r0_aligned_full_s42_v1/status.json
cat outputs/r0_aligned_full_s42_v1/summary.json
```

正式结果无论升降均保留并报告，不按测试反馈搜索种子、超参数或检查点。不能因单次差异断言论文差距根因；还需结合原始PCAP/采样前完整CSV及历史处理记录。找到完整上游文件后，用已有targeted工具固定官方默认流程重算小范围上下文；有确定不一致再决定重建CSV和图，不能用最终平衡CSV补造历史滚动上下文。

## 本地交付与待执行状态

本地只完成Python3.9 AST语法/编译和锁定官方训练函数循环/插入位置静态核查；没有导入模型、运行新增入口或执行数据检查。服务器CUDA冒烟已真实通过；正式对照训练及上游重算仍待实际证据，不能记录为完成。

## 服务器CUDA冒烟回传（2026-10-19）

用户回传已归档至[r0_aligned_smoke_evidence_20261019.json](r0_aligned_smoke_evidence_20261019.json)，未直接取回服务器summary或checkpoint文件。工具identity与交付源码一致，为`082c4dc27605235106a5409aeacb5cd2aa9e9ff5bbcbf59f7099ab609b040080`。

- status=completed，exit_code=0，mode=smoke，epochs_completed=2，optimizer_step_passed=true，test=null。
- Tesla T4/CUDA，Python3.9.23、torch2.3.1+cu118、PyG2.5.3。256个训练样本，每轮4步，总8步；参数431704，首次更新46个参数张量。
- 6个GAT均接收实际边属性，Contain/reverse为634×4，Link为1140×1；Link和reverse关系边投影梯度非零。Contain的零梯度不能单独作为错误证据。
- 两轮训练loss为1.878309428691864和1.2582197487354279，训练accuracy为0.34765625和0.69921875。它们是小样本冒烟指标，不能用于推断正式精度或预计全量耗时。
- last.pt SHA256为`a450f50b30dc52428e85dbba560594e7dcaabc085684397b1fbd626dd8a23217`，总用时7.464秒。运行中出现torch-spline-conv和NVML警告，但本次实际CUDA执行成功，不因警告改动服务器环境。

结论：R0-A已满足其独立CUDA冒烟门槛，可进入一次固定30轮正式对照。正式入口仍会重新校验匹配绑定；尚无正式训练已启动或性能差距已解决的证据。
