# 服务器实验输出分类

新增 `scripts/organize_server_outputs.py`，仅使用标准库。默认预览，Linux 服务器加 `--apply` 创建 `outputs/_organized/` 分类入口与仓库根目录 `XG-NID-organized.code-workspace`。

分类包括 Batch 正式训练及启动记录、原 E0–E4、门控改进、门控诊断、R0、探测核对、恢复快照、历史输出和其他待核实项。每次运行的目录、`.log`、`.pid`、`.supervisor.pid`、`.exitcode`、`.restart.json` 归到同一组。失败启动只有日志而没有结果目录时也会列出；不按文件名推断成功或筛选最佳模型。

原始文件不搬移、不删除、不改名。正在运行的 E0-B512/E3a 可继续写原目录。配置、检查点绑定中的原绝对路径和哈希不变。分类条目是相对符号链接，不复制权重、不额外占用一份模型空间。请勿在链接目录中删除或修改结果，因为操作会作用于原文件。

已有 `.vscode/settings.json` 与旧 `XG-NID-results.code-workspace` 保留。新工作区隐藏代码根目录下的原 `outputs`，在侧边栏按类别显示结果；直接打开普通仓库目录时仍显示原布局。

可重复运行，新增实验会补充链接，索引状态重新读取。已存在链接验证一致后复用，不自动清理失效链接。遇到同名自定义工作区、未托管分类目录、目录被替换成链接或链接目标冲突时拒绝覆盖。

`outputs/_organized/README.md` 是分类索引，`inventory.json` 记录原路径和根级 status.json。磁盘 status 可能陈旧，`running` 不证明进程仍在；本工具不发信号、不更改 GPU 设置、不启动训练，也不检查模型内容。

服务器从交付提交取出脚本即可，不需要切换训练工作树：

```bash
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
git show FETCH_HEAD:scripts/organize_server_outputs.py > /tmp/xgnid_organize_outputs.py
/home/nzxn/miniconda3/envs/wx/bin/python -B /tmp/xgnid_organize_outputs.py --repo . --apply
code -r XG-NID-organized.code-workspace
```

若服务器终端的 `code` 命令不可用，在当前 VS Code 远程连接中使用“文件 → 从文件打开工作区”，选择服务器仓库根目录的新工作区文件。

本地验证：5项标准库测试通过；POSIX 符号链接端到端测试在 Windows 跳过，保留供 Linux 运行。对真实本地 outputs 的只读预览完成，发现65组。尚未执行服务器整理，不将本地预览当作服务器成功记录。
