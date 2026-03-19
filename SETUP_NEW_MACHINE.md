# New Machine Setup — Cursor One-Click Instruction

Copy the entire content of the block below and paste it into Cursor Agent chat
on your new machine. Open an **empty folder** (your new `2026Projects/`) as the
Cursor workspace first.

---

## Cursor Instruction (copy everything between the two `===` lines)

===START===

请在当前工作区（即我的 2026Projects 文件夹）中完成以下所有步骤。当前文件夹就是项目根目录，不要创建额外的子层级。遇到错误时自动重试或换方案，不要停下来问我。用中文回复我。

### 第一步：检测环境

1. 检测操作系统（Windows / Linux / macOS）。
2. 检测 `git` 是否在 PATH 中可用。如果不可用：
   - Windows：用 Python 下载 PortableGit 到 `~/PortableGit`，解压后使用。
   - Linux/macOS：运行 `sudo apt install git` 或 `brew install git`。
3. 检测 `python3`（或 `python`）版本 >= 3.10。如果不可用，提示我安装。
4. 检测 `gh`（GitHub CLI）是否可用。如果不可用：
   - Windows：用 Python 下载 gh CLI zip 到 `~/gh_cli`，解压后使用。
   - Linux：`sudo apt install gh` 或从 GitHub Releases 下载。
   - macOS：`brew install gh`。

### 第二步：GitHub 认证

1. 运行 `gh auth login`，使用 HTTPS 协议、浏览器认证方式。
2. 如果网络不通（超时），提示我提供 Personal Access Token (PAT)，然后用 `echo TOKEN | gh auth login --with-token` 认证。
3. 认证后运行 `gh auth setup-git` 配置 credential helper。
4. 确保 credential helper 指向 `gh` 而不是 `git`（检查 `~/.gitconfig` 里 `credential.https://github.com.helper` 的路径）。

### 第三步：配置 Git

```
git config --global user.email "huwei@amss.ac.cn"
git config --global user.name "Wei Hu"
```

如果直连 GitHub 失败（`git ls-remote` 超时），自动配置代理：
```
git config --global http.proxy socks5h://127.0.0.1:7897
git config --global https.proxy socks5h://127.0.0.1:7897
```
配置后重试。如果代理也不通，尝试 `http://127.0.0.1:7897`。如果都不通，提示我检查网络/代理。

### 第四步：克隆三个项目

在当前工作区根目录下执行：
```
git clone https://github.com/huwei0121/BUPTR.git
git clone https://github.com/huwei0121/DFOETR.git
git clone https://github.com/huwei0121/DisGRem.git
```

### 第五步：为每个项目创建 Python 虚拟环境并安装依赖

对 BUPTR、DFOETR、DisGRem 三个项目分别执行：
```
cd <project>/codes
python -m venv .venv          # Windows 下 python，Linux/macOS 下 python3
# 激活：
#   Windows PowerShell: .venv\Scripts\Activate.ps1
#   Windows cmd:        .venv\Scripts\activate.bat
#   Linux/macOS:        source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 第六步：恢复图片

图片文件不在 Git 中，需要从 CSV 数据重新生成。对每个项目运行其 replot 脚本：

- **BUPTR**: `cd BUPTR/codes && python scripts/replot_from_csv.py`
- **DisGRem**: `cd DisGRem/codes && python scripts/replot.py`
- **DFOETR**: 如果 `codes/scripts/` 下有 replot 脚本就运行，否则跳过。

如果 replot 脚本因缺少可选依赖（如 pdfo、pybobyqa）报错，只需忽略那些导入错误——图片生成只依赖 numpy、matplotlib、pandas。

### 第七步：复制 CONVENTIONS.md 到工作区根目录

从任意一个克隆的项目（如 BUPTR）复制 `CONVENTIONS.md` 到当前工作区根目录：
```
cp BUPTR/CONVENTIONS.md ./CONVENTIONS.md
```

同样复制 `.cursor/rules/`：
```
mkdir -p .cursor/rules   # Windows: New-Item -ItemType Directory -Path .cursor\rules -Force
cp BUPTR/.cursor/rules/project-conventions.mdc .cursor/rules/project-conventions.mdc
```

### 第八步：验证

对每个项目执行以下检查并报告结果：
1. `git status` — 应该是 clean 状态
2. `git remote -v` — 应该指向 `https://github.com/huwei0121/<project>.git`
3. `git pull` — 应该成功（Already up to date）
4. 进入 `codes/` 目录，激活 venv，运行 `python -c "import numpy; print(numpy.__version__)"` — 应该成功

最后给我一个表格汇总三个项目的状态：Git 是否连通、Python 环境是否就绪、图片是否已恢复。

===END===

---

## Notes

- The instruction is designed to work on Windows, Linux, and macOS.
- Proxy settings (`socks5h://127.0.0.1:7897`) assume Clash/V2Ray is running.
  If your new machine uses a different proxy port, modify step 3 accordingly.
- If `gh auth login --web` fails due to network, you'll need a GitHub PAT from
  https://github.com/settings/tokens (scope: `repo`).
- Local directory names for the repos will be `BUPTR/`, `DFOETR/`, `DisGRem/`
  (matching the GitHub repo names, no version suffix).
