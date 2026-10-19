# 训练期间整理 outputs 展示

截图包含R0-A当前正式训练、原R0、E0/E1、冒烟、核查和历史结果及日志/PID。它们不全是临时文件；文件小并不代表无用。当前R0-A是否已正常进入训练仍需日志确认，不能因截图中目录存在就记为完成。

整理方式为创建独立VS Code工作区，按用途列出入口，代码根目录收起重复的outputs展示。原outputs下的文件名、路径、内容和已存在的.vscode/settings.json不修改；没有物理搬移、删除或磁盘空间回收。该方式避免打断后台训练，也保留后续脚本对固定冒烟/结果/launch路径的引用。不会把需要保留的证据当成垃圾清理。

在服务器原仓库wx环境运行：

```bash
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
git show origin/codex/r0-edge-preparation:scripts/organize_outputs_view.py > /tmp/xgnid_organize_outputs.py
python -B /tmp/xgnid_organize_outputs.py --repo .
```

工具仅使用标准库，读取目录名称，生成仓库根目录的`XG-NID-results.code-workspace`。不读模型/CSV/图数据，不启动或结束任何训练。已存在且内容相同则保持不变；若已有工作区被自定义或目录清单改变则拒绝覆盖。没有自动清理步骤。

之后在当前服务器的VS Code远程窗口选择“文件 → 从文件打开工作区”，打开该文件。若当前终端支持VS Code远程CLI，也可执行：

```bash
code XG-NID-results.code-workspace
```

侧栏按顺序显示代码、R0-A正式训练、R0-A冒烟、原R0、E0/E1（以及实际已存在的E2/E3）、全部产物。把“90 全部产物与日志”保持折叠即可减少侧栏杂乱。日志没有被隐藏或删除，需要时展开该入口，或继续使用原tail命令。进入正式训练目录可查看epochs/status；相邻的.log/.pid/.exitcode仍在全部产物入口。

本地仅做源码语法检查，服务器工作区生成和界面打开待执行。此工作区不改变已有复现实验协议。需要物理归档时，应在运行结束后按具体目录确认引用关系，再单独实施，不能现在批量移动原R0工作树、清单、checkpoint或冒烟证据。
