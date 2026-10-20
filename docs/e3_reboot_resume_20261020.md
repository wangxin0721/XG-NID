# E3 重启暂停与独立续训交付（2026-10-20）

用户明确要求立即保存检查点、重启服务器后继续 E3。未在本地执行模型、检查点反序列化、暂停进程或训练；服务器操作由用户执行。冻结 `scripts/train_grouped.py`、模型、划分以及原 E0～E3 文件保持不变。

## 暂停实际回传

修正版工具提交 `dbbef8a` 已在服务器执行成功，用户回传：

- `status=snapshot_verified_process_stopped`；PID 268759 已 SIGSTOP，未自动重启服务器，也未停止其他任务。
- 原目录：`/home/nzxn/下载/WX/load/XG-NID/outputs/grouped_full_E3_s42_v1`。
- 快照：`/home/nzxn/下载/WX/load/XG-NID/outputs/e3_restart_snapshot_20261020T103302548182Z`。
- 指针：`outputs/grouped_full_E3_s42_v1.restart.json`。
- 保存第58轮、最佳第48轮、连续未改善10轮、总预算仍为100轮。若当前未保存的第59轮已开始，其进度不包含在检查点内。
- 时间：2026-10-20 18:33:04 北京时间（UTC 10:33:04）。

已在服务器复制、比较 SHA、CPU 加载可信检查点并执行 fsync/sync。四个快照文件 SHA：

| 文件 | SHA256 |
|---|---|
| config.json | `2dd9e634ac9895ccd39223665d6b315e3b7e7611b41efb45967de6ba6d41bff3` |
| last.pt | `073d89edfc5da62f792b6ee2910c741e892534f9ff31b8c9e2aab33fe6f87ba1` |
| best.pt | `600c4f2c3f37c692c920b8001843a92e3d05469c2f09b88d9a4f9b707549a14d` |
| epochs.jsonl | `795380fdc68bd57f8a47e5df44f1716eb1ae20afddf12bd2f9fdb31e1bd0942f` |

初版 `13561a9` 暂停后因配置比较方式错误未通过校验：torch 检查点保留类别计数的整数键，JSON 文件则将它们转成字符串键。`dbbef8a` 改为 JSON 规范化后比较，并增加明确的缺失字段/差异字段错误。已收到修正版成功回传；初次失败留下的独立副本没有覆盖原文件。恢复工具同样执行规范化比较，不能通过忽略实际字段差异来绕过校验。

本记录证明 E3 暂停和快照保存成功，不证明用户已执行服务器重启，也不证明后续恢复成功。重启会影响服务器所有任务，本工具只备份这次 E3。

## 独立恢复机制

新增 `scripts/resume_e3_server.py`。它读取上述指针，不选择最新目录、不按得分选快照；逐项校验快照 SHA、原训练源码 SHA、配置、torch/PyG 版本、历史 epoch、最佳验证选模和检查点内容。只有明确指定 `--resume` 才运行训练；`--check-only` 只做 CPU 检查和源码 AST 编译，不运行前向或反向。

原冻结入口没有 resume 参数。新工具在新进程内生成原 `train_one` 的 AST 副本，仅加入六处严格匹配的恢复钩子：原 config、模型/优化器/采样器/RNG/历史恢复、剩余轮次范围、epoch 日志追加、最终指标的恢复说明、最终测试访问记录。每种钩子必须恰好匹配一次，否则拒绝执行。原训练循环体、验证、严格改善选模、检查点内容和最终测试实现不变，不修改冻结源文件。

恢复 model_state（含 BN 状态）、Adam optimizer_state、sampler_rng_state、torch_rng_state、所有已保存的可见 CUDA RNG 状态；按第58轮继续第59轮，沿用最佳第48轮和 stale=10。若第59/60轮不改善，会在第60轮停止；不重设100轮新预算，不重新早停计数。若保存时已达到早停/总预算，则跳过额外训练，仅完成原最佳权重的最终测试。

历史日志只接受检查点对应的完整前缀；日志先写、检查点后写造成的未保存尾部会明确列为 discarded 并重跑该轮。暂停恰好发生在 last 更新而 best 未更新时，只在 last 确为当前最佳的情况下使用 last 修复新目录的 best，不改原文件。

新结果目录为 `outputs/grouped_full_E3_s42_resume_v1`，模型产物仍在其 `E3/` 子目录。原配置字节 SHA 必须保持一致；原实验和快照保持只读。恢复说明写入新目录 `restart.json` 与最终 metrics，明确运行时间仅指恢复会话，不含重启前或关机时间。原目录的旧 running 状态不会被伪装成新任务状态，后续必须读取新目录。

恢复时复用原报告路径，并再次调用原入口的 readiness 校验 E0/E1 预跑、E0 基线、E2 完整报告、配对冒烟、split 和训练控制；硬件型号及可见 CUDA 设备数量也须与旧配置一致。发现原 E3 仍存在（包括已停止状态）或目标续训进程存在时拒绝重复运行；不自动杀进程、重试、安装依赖或更换模型。

不能保证逐位一致：旧检查点没有 Python/NumPy/worker RNG 或完整耗时，重启后 worker 重新创建、CUDA 可能非确定。现有图 loader 无随机数据增强，独立采样器的下一轮顺序可以恢复；仍需以实际服务器回传为准，不能将“已写入口”标为“已恢复成功”。若原任务在暂停前已经开始最终测试，恢复可能重复同一固定最佳权重的推断，须记为恢复重复访问，不进行测试驱动的参数选择。

## 重启后执行

不要再执行原 E3 启动命令，也不要再次运行 pause。先确认 GPU0 可以供本任务使用，激活原 wx 环境并获取新增工具；只 fetch/git show，不切换分支、不 pull、不更改冻结文件。下面分支引用在交付时以最终提交号固定。

```bash
. /home/nzxn/miniconda3/etc/profile.d/conda.sh && conda activate wx
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
git show origin/codex/r0-edge-preparation:scripts/resume_e3_server.py > /tmp/xgnid_resume_e3.py
python -B /tmp/xgnid_resume_e3.py --repo . --check-only
```

只有 check-only 返回 `checkpoint_checks_passed`，并显示 saved_epoch=58、next_epoch=59、best_epoch=48、stale_epochs=10 后，才执行一次：

```bash
if [ -e outputs/grouped_full_E3_s42_resume_v1 ] || [ -e outputs/grouped_full_E3_s42_resume_v1.log ]; then
    echo "恢复目录或日志已存在，先核查，不重复提交。"
else
    CUDA_VISIBLE_DEVICES=0 nohup python -B -u /tmp/xgnid_resume_e3.py --repo . --resume >> outputs/grouped_full_E3_s42_resume_v1.log 2>&1 < /dev/null &
    echo "$!" > outputs/grouped_full_E3_s42_resume_v1.pid
    echo "恢复请求已提交；请检查日志确认实际恢复。"
fi
tail -n 30 -F outputs/grouped_full_E3_s42_resume_v1.log
```

`checkpoint_checks_passed` 仅证明前置检查通过；出现 `Restored E3 epoch=58, best_epoch=48, stale=10; budget unchanged` 才说明模型/优化器/采样器/随机状态恢复调用已完成，随后第59轮指标才证明训练继续推进。若恢复后的 CUDA 不可用、可见设备数/型号不符、旧进程仍存在或文件校验失败，保留错误，不回退 CPU、不覆盖目录。

完成查询：

```bash
cat outputs/grouped_full_E3_s42_resume_v1/status.json
cat outputs/grouped_full_E3_s42_resume_v1/E3/metrics.json
```

若新目录已存在，先查看其状态和错误，不删除后重复执行；只有明确核查失败原因后，才为另一次恢复尝试显式选择新的输出目录。工具无自动恢复重试。暂停前保存第58轮与后续新目录应合并解释为一次有恢复记录的实验，不算新增 seed 或独立重复实验。

## 本次验证范围

暂停：已收到真实服务器快照校验与进程暂停成功证据。恢复：Python 3.9 语法检查通过；六处恢复钩子各恰好匹配一次；生成的函数可编译，恢复前后 epoch 循环体 AST 完全相同；整数键规范化与58/48/10轮次边界检查通过。只执行了标准库源码转换验证，未调用生成的训练函数，未在本地执行 checkpoint 加载、模型构建或训练。服务器 check-only、真实恢复及后续训练结果均待回传。
