# E1运行期间准备R0源数据清单

执行条件：已回传E1主/子进程和17轮日志；环境wx、原图目录已实测存在。此操作只读源CSV/图，会有CPU/I/O开销；使用低CPU优先级，有ionice时使用idle I/O级别。无GPU任务，不重启或替换现有环境。不得在原图目录仍被改写时运行。

在服务器**原始仓库根目录**的已有wx终端复制以下整段。不要把提示符一起复制。git worktree只新建独立代码目录，不改变E1当前分支；每次使用新的时间戳，不能复用已存在输出。

```bash
git fetch origin codex/r0-edge-preparation
R0_SOURCE="$PWD"
R0_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
R0_CODE="$R0_SOURCE/outputs/r0_code_$R0_STAMP"
R0_PREP="$R0_SOURCE/outputs/r0_prepare_$R0_STAMP"
git worktree add --detach "$R0_CODE" origin/codex/r0-edge-preparation
```

确认worktree创建成功，再执行以下后台命令。脚本将自身PID写入status；shell的启动PID也另存。全程只加载CPU图，不做模型构建、反向或优化步骤。

```bash
if command -v ionice >/dev/null 2>&1; then
  nohup env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ionice -c 3 nice -n 10 \
    /home/nzxn/miniconda3/envs/wx/bin/python -u "$R0_CODE/scripts/prepare_r0_server.py" \
    --source-repo "$R0_SOURCE" --output "$R0_PREP" > "$R0_PREP.log" 2>&1 < /dev/null &
else
  nohup env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 \
    /home/nzxn/miniconda3/envs/wx/bin/python -u "$R0_CODE/scripts/prepare_r0_server.py" \
    --source-repo "$R0_SOURCE" --output "$R0_PREP" > "$R0_PREP.log" 2>&1 < /dev/null &
fi
printf '%s\n' "$!" > "$R0_PREP.pid"
printf 'R0_PREP=%s\nR0_CODE=%s\n' "$R0_PREP" "$R0_CODE"
```

此处后台命令结束返回提示符不等于扫描完成。扫描185557图耗时以实际为准，日志每2000图打印进度；I/O idle时可能等待更久，不据此判定卡死。若ionice存在但启动被系统拒绝，回传日志，不自动绕过或反复启动。

```bash
tail -n 30 "$R0_PREP.log"
test ! -f "$R0_PREP/status.json" || cat "$R0_PREP/status.json"
test ! -f "$R0_PREP/summary.json" || cat "$R0_PREP/summary.json"
```

回传R0_PREP真实路径、日志和status；completed后再回传summary.json及source_evidence.json。失败则error.log；不用上传大型清单或图。进入新终端后先把R0_PREP赋为启动时打印的实际路径，不要生成新的时间戳代替已有运行目录。

只有源图逐类数量、内容/标签、重叠和来源限制经核查后，才准备下一步GPU smoke。R0正式训练仍未启动，不能自动串联。worktree保留以供后续相同代码hash/runtime的smoke/full使用，当前不删除。
