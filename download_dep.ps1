#Requires -Version 5.1
<#
.SYNOPSIS
    从 Git 仓库下载并设置项目依赖项。
.DESCRIPTION
    此脚本确保 'lib' 目录存在，在其中创建一个 '__init__.py' 文件，
    然后克隆指定的 Git 仓库。它处理克隆后需要重命名仓库
    或仅需要克隆仓库中某个子目录的情况。
    脚本会检查现有目录以避免冗余操作。
.NOTES
    作者: Gemini Code Assist
    请确保 Git 已安装并在您的 PATH 环境变量中可访问。
    对于通过 SSH (git@github.com:...) 克隆的仓库，必须配置好 SSH 密钥。
#>

# 设置严格模式以进行更好的错误检查
Set-StrictMode -Version Latest
# 遇到任何错误即停止脚本执行
$ErrorActionPreference = "Stop"

# 确定脚本的目录，以便路径相对于脚本位置
# 如果在 ISE 中运行，$PSScriptRoot 可能为空，则回退到当前位置。
if ($PSScriptRoot) {
    $scriptDir = $PSScriptRoot
}
else {
    $scriptDir = (Get-Location -PSProvider FileSystem).Path
}
Write-Verbose "脚本目录: $scriptDir"

# 定义相对于脚本目录的基础库路径
$libBaseName = "lib"
$libDirPath = Join-Path -Path $scriptDir -ChildPath $libBaseName

# --- 1. 确保 'lib' 目录存在 ---
if (-not (Test-Path -Path $libDirPath -PathType Container)) {
    Write-Host "创建目录: $libDirPath"
    New-Item -ItemType Directory -Path $libDirPath -Force | Out-Null
}
else {
    Write-Host "目录 $libDirPath 已存在。"
}

# --- 2. 如果 'lib/__init__.py' 不存在，则创建它 ---
$initPyFileName = "__init__.py"
$initPyPath = Join-Path -Path $libDirPath -ChildPath $initPyFileName
if (Test-Path -Path $initPyPath -PathType Leaf) {
    # 'Leaf' 表示文件
    Write-Host "文件 $initPyPath 已经存在，跳过创建。"
}
else {
    Write-Host "创建文件: $initPyPath"
    New-Item -ItemType File -Path $initPyPath -Force | Out-Null
}

# --- 3. 定义并处理依赖项 ---
$dependencies = @(
    @{
        TargetDirNameInLib = "MuxkitTools" # 'lib' 目录中最终的目录名
        RepoUrl            = "git@github.com:LinhMuks-DFox/Muxkit.DeepLearning.Tools.git"
        ClonedRepoName     = "Muxkit.DeepLearning.Tools" # 'git clone' 创建的目录名
        SourceSubDir       = $null # 没有子目录，移动整个 ClonedRepoName
    },
    @{
        TargetDirNameInLib = "AudioSet"
        RepoUrl            = "https://github.com/LinhMuks-DFox/MuxKit.AudioSet.IO.git"
        ClonedRepoName     = "MuxKit.AudioSet.IO"
        SourceSubDir       = "AudioSet" # 移动 MuxKit.AudioSet.IO/AudioSet
    },
    @{
        TargetDirNameInLib = "esc50_io"
        RepoUrl            = "git@github.com:LinhMuks-DFox/esc-50-io.git"
        ClonedRepoName     = "esc-50-io"
        SourceSubDir       = $null # 没有子目录，移动整个 ClonedRepoName
    }
)

foreach ($dep in $dependencies) {
    $targetPathInLib = Join-Path -Path $libDirPath -ChildPath $dep.TargetDirNameInLib
    Write-Host ("-" * 60)
    Write-Host "正在处理依赖: $($dep.TargetDirNameInLib)"

    if (Test-Path -Path $targetPathInLib -PathType Container) {
        Write-Host "[?] $($dep.TargetDirNameInLib) 已存在于 $targetPathInLib，跳过克隆。"
        continue
    }

    # 'git clone' 将放置仓库的路径 (在脚本的目录下)
    $clonedRepoPathAtScriptDir = Join-Path -Path $scriptDir -ChildPath $dep.ClonedRepoName

    # 清理先前失败的克隆尝试可能留下的任何目录
    if (Test-Path -Path $clonedRepoPathAtScriptDir -PathType Container) {
        Write-Warning "发现先前存在的临时克隆目录 $clonedRepoPathAtScriptDir，将移除。"
        Remove-Item -Path $clonedRepoPathAtScriptDir -Recurse -Force
    }

    Write-Host "正在克隆 $($dep.RepoUrl) 到 $clonedRepoPathAtScriptDir..."
    try {
        # 执行 git clone。输出默认不被捕获，除非重定向。
        # 我们在其后立即检查 $LASTEXITCODE。
        git clone $($dep.RepoUrl) $clonedRepoPathAtScriptDir
        if ($LASTEXITCODE -ne 0) {
            # 如果 $ErrorActionPreference 是 Stop，这个自定义错误消息将被外部的 try/catch 捕获
            throw "Git 克隆 $($dep.RepoUrl) 失败。退出码: $LASTEXITCODE"
        }
        Write-Host "克隆 $($dep.ClonedRepoName) 成功。"

        $sourceToMove = $clonedRepoPathAtScriptDir
        if ($null -ne $dep.SourceSubDir -and -not ([string]::IsNullOrWhiteSpace($dep.SourceSubDir))) {
            $sourceToMove = Join-Path -Path $clonedRepoPathAtScriptDir -ChildPath $dep.SourceSubDir
            if (-not (Test-Path -Path $sourceToMove)) {
                # 检查它是文件还是目录
                throw "克隆的仓库 $($dep.ClonedRepoName) 中未找到指定的源子路径: $($dep.SourceSubDir)"
            }
        }

        Write-Host "正在移动 $sourceToMove 到 $targetPathInLib..."
        Move-Item -Path $sourceToMove -Destination $targetPathInLib -Force

        # 如果移动了子目录 (指定了 SourceSubDir)，
        # 并且父克隆目录 ($clonedRepoPathAtScriptDir) 仍然存在，则将其移除。
        if (($null -ne $dep.SourceSubDir -and -not ([string]::IsNullOrWhiteSpace($dep.SourceSubDir))) -and `
            (Test-Path -Path $clonedRepoPathAtScriptDir -PathType Container)) {
            Write-Host "正在清理临时克隆目录: $clonedRepoPathAtScriptDir"
            Remove-Item -Path $clonedRepoPathAtScriptDir -Recurse -Force
        }
        # 如果整个仓库被移动并重命名 (例如 Muxkit.DeepLearning.Tools 到 lib/MuxkitTools)，
        # $clonedRepoPathAtScriptDir 是 Move-Item 的源，并且在其原始位置不再存在。
        # 因此，在这种情况下不需要显式清理 $clonedRepoPathAtScriptDir。

        Write-Host "[?] $($dep.TargetDirNameInLib) 成功安装到 $targetPathInLib。"

    }
    catch {
        Write-Error "处理依赖 $($dep.TargetDirNameInLib) 时发生错误: $($_.Exception.Message)"
        # 如果存在部分克隆的目录，则进行清理，因为脚本将停止。
        if (Test-Path -Path $clonedRepoPathAtScriptDir -PathType Container) {
            Write-Warning "发生错误，尝试清理 $clonedRepoPathAtScriptDir..."
            Remove-Item -Path $clonedRepoPathAtScriptDir -Recurse -Force -ErrorAction SilentlyContinue
        }
        # 重新抛出异常以确保脚本根据 $ErrorActionPreference = "Stop" 终止
        throw
    }
}

Write-Host ("-" * 60)
Write-Host "所有依赖下载和设置已完成。"
