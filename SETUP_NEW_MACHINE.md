# New Machine Setup — One-Click Project Recovery

Copy the entire instruction block below and paste it into **Cursor Agent chat** on
your new machine. Open an **empty folder** (your new `2026Projects/`) as the Cursor
workspace first.

This instruction also works as a manual checklist for human execution.

---

## Cursor / Human Instruction

===START===

请在当前工作区（即我的 2026Projects 文件夹）中完成以下所有步骤。当前文件夹就是项目根目录，不要创建额外的子层级。遇到错误时自动重试或换方案，不要停下来问我。用中文回复我。

### 第一步：检测环境

1. 检测操作系统（Windows / Linux / macOS）。
2. 检测 `git` 是否在 PATH 中可用。如果不可用：
   - **Windows**：用 Python 下载 PortableGit 到 `~/PortableGit`，解压后将其 `cmd` 目录加入 PATH。
     下载地址: `https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.2/PortableGit-2.47.1.2-64-bit.7z.exe`
     下载命令：`python -c "import urllib.request; urllib.request.urlretrieve('...url...', r'%USERPROFILE%\PortableGit.exe')"`
     解压命令：`%USERPROFILE%\PortableGit.exe -y -o%USERPROFILE%\PortableGit`
   - **Linux**: `sudo apt update && sudo apt install -y git`
   - **macOS**: `brew install git`
3. 检测 `python3`（或 Windows 上的 `python`）版本 >= 3.10。如果不可用，提示安装。
4. 检测 `gh`（GitHub CLI）是否可用。如果不可用：
   - **Windows**：用 Python 下载 gh CLI zip 到 `~/gh_cli`，解压后将其 `bin` 目录加入 PATH。
     下载地址: `https://github.com/cli/cli/releases/download/v2.63.2/gh_2.63.2_windows_amd64.zip`
   - **Linux**: `sudo apt install -y gh` 或从 GitHub Releases 下载。
   - **macOS**: `brew install gh`

### 第二步：GitHub 认证

使用 PAT (Personal Access Token) 直接认证：

```bash
echo "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" | gh auth login --hostname github.com --git-protocol https --with-token
```

如果上述命令因网络超时失败，先配置代理（见第三步），再重试。

认证后配置 credential helper：

```bash
gh auth setup-git
```

**重要**：检查 `~/.gitconfig`（Windows: `%USERPROFILE%\.gitconfig`），确认
`credential.https://github.com.helper` 指向 `gh`（而非 `git`）。如果看到路径包含
`git.exe auth git-credential`，请将其改为 `gh.exe auth git-credential`。

### 第三步：配置 Git

```bash
git config --global user.email "huwei@amss.ac.cn"
git config --global user.name "Wei Hu"
```

测试连通性：

```bash
git ls-remote https://github.com/huwei0121/BUPTR.git
```

如果超时，依次尝试以下代理配置：

```bash
# 方案 1：SOCKS5 代理（Clash / V2Ray 常用端口）
git config --global http.proxy  socks5h://127.0.0.1:7897
git config --global https.proxy socks5h://127.0.0.1:7897

# 方案 2：HTTP 代理
git config --global http.proxy  http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897

# 方案 3：直连（如果以上都不行，取消代理试试）
git config --global --unset http.proxy
git config --global --unset https.proxy
```

配置后重新测试 `git ls-remote`，直到成功为止。

### 第四步：克隆四个项目

在当前工作区根目录下执行：

```bash
git clone https://github.com/huwei0121/BUPTR.git
git clone https://github.com/huwei0121/DFOETR.git
git clone https://github.com/huwei0121/DisGRem.git
git clone https://github.com/huwei0121/RSSM.git
```

### 第五步：为每个项目创建 Python 虚拟环境并安装依赖

对 BUPTR、DFOETR、DisGRem、RSSM 四个项目分别执行：

```bash
cd <project>/codes
python -m venv .venv          # Windows 下用 python，Linux/macOS 用 python3

# 激活虚拟环境：
#   Windows PowerShell:  .venv\Scripts\Activate.ps1
#   Windows cmd:         .venv\Scripts\activate.bat
#   Linux / macOS:       source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

RSSM 的 `codes/` 目前只有骨架（数值实验尚未开始），但仍需安装依赖以备后用。

### 第六步：恢复图片

图片文件不在 Git 中，需要从 CSV 数据重新生成。对每个项目运行其 replot 脚本：

- **BUPTR**: `cd BUPTR/codes && python scripts/replot_from_csv.py`
- **DisGRem**: `cd DisGRem/codes && python scripts/replot.py`
- **DFOETR**: 如果 `codes/scripts/` 下有 replot 脚本就运行，否则跳过。
- **RSSM**: 跳过（尚无数值实验数据）。

如果 replot 脚本因缺少可选依赖（如 pdfo、pybobyqa）报错，忽略那些导入错误——
图片生成只依赖 numpy、matplotlib、pandas。

### 第七步：复制共享文件到工作区根目录

```bash
# Linux / macOS
cp BUPTR/CONVENTIONS.md ./CONVENTIONS.md
cp BUPTR/SETUP_NEW_MACHINE.md ./SETUP_NEW_MACHINE.md
mkdir -p .cursor/rules
cp BUPTR/.cursor/rules/project-conventions.mdc .cursor/rules/project-conventions.mdc

# Windows PowerShell
Copy-Item BUPTR\CONVENTIONS.md .\CONVENTIONS.md
Copy-Item BUPTR\SETUP_NEW_MACHINE.md .\SETUP_NEW_MACHINE.md
New-Item -ItemType Directory -Path .cursor\rules -Force
Copy-Item BUPTR\.cursor\rules\project-conventions.mdc .cursor\rules\project-conventions.mdc
```

### 第八步：验证

对每个项目执行以下检查并报告结果：

1. `git status` — 应该是 clean 状态
2. `git remote -v` — 应该指向 `https://github.com/huwei0121/<project>.git`
3. `git pull` — 应该成功（Already up to date）
4. 进入 `codes/` 目录，激活 venv，运行 `python -c "import numpy; print(numpy.__version__)"` — 应该成功

最后给我一个表格汇总四个项目的状态：Git 是否连通、Python 环境是否就绪、图片是否已恢复。

===END===

---

## Notes

- **Security**: This file contains a GitHub PAT. Share only with trusted collaborators.
  If the token is compromised, revoke it at https://github.com/settings/tokens and
  generate a new one.
- **Proxy**: The SOCKS5 proxy `127.0.0.1:7897` assumes Clash/V2Ray is running locally.
  Adjust the port if your proxy software uses a different one.
- **Projects**: There are 4 repositories:
  - `BUPTR` — Trust-region optimization solver
  - `DFOETR` — Derivative-free optimization with ellipsoidal trust region
  - `DisGRem` — Distributed gradient-regularized Newton method
  - `RSSM` — Regular simplicial search method (theory complete, numerics pending)
- **Local directory names** will match GitHub repo names: `BUPTR/`, `DFOETR/`,
  `DisGRem/`, `RSSM/`.
- **After setup**, daily workflow is:
  ```bash
  cd <project>
  git pull                    # sync latest
  # ... work ...
  git add -A && git commit -m "message" && git push
  ```
