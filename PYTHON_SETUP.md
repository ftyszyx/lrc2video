# 🔧 Python 3.10+ 安装指南

## ⚠️ 当前问题

你的系统只有 **Python 2.7**，但项目需要 **Python 3.10+**。

## 📥 安装 Python 3.10+

### 方法 1: 从官方网站下载（推荐）

1. 访问 https://www.python.org/downloads/
2. 下载 **Python 3.12** 或 **Python 3.11** 的 Windows 安装程序
3. 运行安装程序，**重要**：勾选 "Add Python to PATH"
4. 完成安装

### 方法 2: 使用 Chocolatey

```powershell
# 如果已安装 Chocolatey
choco install python

# 或指定版本
choco install python --version=3.12.0
```

### 方法 3: 使用 Windows Package Manager (winget)

```powershell
winget install Python.Python.3.12
```

## ✅ 验证安装

安装完成后，打开新的 PowerShell 窗口，运行：

```powershell
python --version
```

应该显示 `Python 3.10.x` 或更高版本。

## 🔄 重新创建虚拟环境

安装 Python 3.10+ 后，删除旧的虚拟环境并重新创建：

```powershell
# 1. 删除旧的虚拟环境
rmdir /s /q .venv

# 2. 创建新的虚拟环境
python -m venv .venv

# 3. 激活虚拟环境
.venv\Scripts\activate

# 4. 安装依赖（使用 uv）
uv pip install -e .

# 5. （可选）安装 CUDA 版 PyTorch（需要 NVIDIA GPU）
# 示例以 cu121 为例：
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

## 🚀 启动应用

```powershell
python app.py
```

---

## 📌 常见问题

### Q: 如何卸载 Python 2.7？
A: 
1. 打开 "控制面板" → "程序和功能"
2. 找到 "Python 2.7"
3. 点击卸载

### Q: 安装后 python 命令仍然指向 Python 2.7？
A: 
1. 重启 PowerShell 或 CMD
2. 或使用完整路径：`C:\Python312\python.exe --version`

### Q: 如何同时保留 Python 2.7 和 3.x？
A: 可以同时安装，使用 `py` 命令选择版本：
```powershell
py -2 --version  # Python 2.7
py -3 --version  # Python 3.x
py -3.12 --version  # 指定 Python 3.12
```

---

**安装完 Python 3.10+ 后，请重新运行上面的虚拟环境创建步骤！**
