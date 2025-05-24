#Requires -Version 5.1
<#
.SYNOPSIS
    �� Git �ֿ����ز�������Ŀ�����
.DESCRIPTION
    �˽ű�ȷ�� 'lib' Ŀ¼���ڣ������д���һ�� '__init__.py' �ļ���
    Ȼ���¡ָ���� Git �ֿ⡣�������¡����Ҫ�������ֿ�
    �����Ҫ��¡�ֿ���ĳ����Ŀ¼�������
    �ű���������Ŀ¼�Ա������������
.NOTES
    ����: Gemini Code Assist
    ��ȷ�� Git �Ѱ�װ�������� PATH ���������пɷ��ʡ�
    ����ͨ�� SSH (git@github.com:...) ��¡�Ĳֿ⣬�������ú� SSH ��Կ��
#>

# �����ϸ�ģʽ�Խ��и��õĴ�����
Set-StrictMode -Version Latest
# �����κδ���ֹͣ�ű�ִ��
$ErrorActionPreference = "Stop"

# ȷ���ű���Ŀ¼���Ա�·������ڽű�λ��
# ����� ISE �����У�$PSScriptRoot ����Ϊ�գ�����˵���ǰλ�á�
if ($PSScriptRoot) {
    $scriptDir = $PSScriptRoot
}
else {
    $scriptDir = (Get-Location -PSProvider FileSystem).Path
}
Write-Verbose "�ű�Ŀ¼: $scriptDir"

# ��������ڽű�Ŀ¼�Ļ�����·��
$libBaseName = "lib"
$libDirPath = Join-Path -Path $scriptDir -ChildPath $libBaseName

# --- 1. ȷ�� 'lib' Ŀ¼���� ---
if (-not (Test-Path -Path $libDirPath -PathType Container)) {
    Write-Host "����Ŀ¼: $libDirPath"
    New-Item -ItemType Directory -Path $libDirPath -Force | Out-Null
}
else {
    Write-Host "Ŀ¼ $libDirPath �Ѵ��ڡ�"
}

# --- 2. ��� 'lib/__init__.py' �����ڣ��򴴽��� ---
$initPyFileName = "__init__.py"
$initPyPath = Join-Path -Path $libDirPath -ChildPath $initPyFileName
if (Test-Path -Path $initPyPath -PathType Leaf) {
    # 'Leaf' ��ʾ�ļ�
    Write-Host "�ļ� $initPyPath �Ѿ����ڣ�����������"
}
else {
    Write-Host "�����ļ�: $initPyPath"
    New-Item -ItemType File -Path $initPyPath -Force | Out-Null
}

# --- 3. ���岢���������� ---
$dependencies = @(
    @{
        TargetDirNameInLib = "MuxkitTools" # 'lib' Ŀ¼�����յ�Ŀ¼��
        RepoUrl            = "git@github.com:LinhMuks-DFox/Muxkit.DeepLearning.Tools.git"
        ClonedRepoName     = "Muxkit.DeepLearning.Tools" # 'git clone' ������Ŀ¼��
        SourceSubDir       = $null # û����Ŀ¼���ƶ����� ClonedRepoName
    },
    @{
        TargetDirNameInLib = "AudioSet"
        RepoUrl            = "https://github.com/LinhMuks-DFox/MuxKit.AudioSet.IO.git"
        ClonedRepoName     = "MuxKit.AudioSet.IO"
        SourceSubDir       = "AudioSet" # �ƶ� MuxKit.AudioSet.IO/AudioSet
    },
    @{
        TargetDirNameInLib = "esc50_io"
        RepoUrl            = "git@github.com:LinhMuks-DFox/esc-50-io.git"
        ClonedRepoName     = "esc-50-io"
        SourceSubDir       = $null # û����Ŀ¼���ƶ����� ClonedRepoName
    }
)

foreach ($dep in $dependencies) {
    $targetPathInLib = Join-Path -Path $libDirPath -ChildPath $dep.TargetDirNameInLib
    Write-Host ("-" * 60)
    Write-Host "���ڴ�������: $($dep.TargetDirNameInLib)"

    if (Test-Path -Path $targetPathInLib -PathType Container) {
        Write-Host "[?] $($dep.TargetDirNameInLib) �Ѵ����� $targetPathInLib��������¡��"
        continue
    }

    # 'git clone' �����òֿ��·�� (�ڽű���Ŀ¼��)
    $clonedRepoPathAtScriptDir = Join-Path -Path $scriptDir -ChildPath $dep.ClonedRepoName

    # ������ǰʧ�ܵĿ�¡���Կ������µ��κ�Ŀ¼
    if (Test-Path -Path $clonedRepoPathAtScriptDir -PathType Container) {
        Write-Warning "������ǰ���ڵ���ʱ��¡Ŀ¼ $clonedRepoPathAtScriptDir�����Ƴ���"
        Remove-Item -Path $clonedRepoPathAtScriptDir -Recurse -Force
    }

    Write-Host "���ڿ�¡ $($dep.RepoUrl) �� $clonedRepoPathAtScriptDir..."
    try {
        # ִ�� git clone�����Ĭ�ϲ������񣬳����ض���
        # ���������������� $LASTEXITCODE��
        git clone $($dep.RepoUrl) $clonedRepoPathAtScriptDir
        if ($LASTEXITCODE -ne 0) {
            # ��� $ErrorActionPreference �� Stop������Զ��������Ϣ�����ⲿ�� try/catch ����
            throw "Git ��¡ $($dep.RepoUrl) ʧ�ܡ��˳���: $LASTEXITCODE"
        }
        Write-Host "��¡ $($dep.ClonedRepoName) �ɹ���"

        $sourceToMove = $clonedRepoPathAtScriptDir
        if ($null -ne $dep.SourceSubDir -and -not ([string]::IsNullOrWhiteSpace($dep.SourceSubDir))) {
            $sourceToMove = Join-Path -Path $clonedRepoPathAtScriptDir -ChildPath $dep.SourceSubDir
            if (-not (Test-Path -Path $sourceToMove)) {
                # ��������ļ�����Ŀ¼
                throw "��¡�Ĳֿ� $($dep.ClonedRepoName) ��δ�ҵ�ָ����Դ��·��: $($dep.SourceSubDir)"
            }
        }

        Write-Host "�����ƶ� $sourceToMove �� $targetPathInLib..."
        Move-Item -Path $sourceToMove -Destination $targetPathInLib -Force

        # ����ƶ�����Ŀ¼ (ָ���� SourceSubDir)��
        # ���Ҹ���¡Ŀ¼ ($clonedRepoPathAtScriptDir) ��Ȼ���ڣ������Ƴ���
        if (($null -ne $dep.SourceSubDir -and -not ([string]::IsNullOrWhiteSpace($dep.SourceSubDir))) -and `
            (Test-Path -Path $clonedRepoPathAtScriptDir -PathType Container)) {
            Write-Host "����������ʱ��¡Ŀ¼: $clonedRepoPathAtScriptDir"
            Remove-Item -Path $clonedRepoPathAtScriptDir -Recurse -Force
        }
        # ��������ֿⱻ�ƶ��������� (���� Muxkit.DeepLearning.Tools �� lib/MuxkitTools)��
        # $clonedRepoPathAtScriptDir �� Move-Item ��Դ����������ԭʼλ�ò��ٴ��ڡ�
        # ��ˣ�����������²���Ҫ��ʽ���� $clonedRepoPathAtScriptDir��

        Write-Host "[?] $($dep.TargetDirNameInLib) �ɹ���װ�� $targetPathInLib��"

    }
    catch {
        Write-Error "�������� $($dep.TargetDirNameInLib) ʱ��������: $($_.Exception.Message)"
        # ������ڲ��ֿ�¡��Ŀ¼�������������Ϊ�ű���ֹͣ��
        if (Test-Path -Path $clonedRepoPathAtScriptDir -PathType Container) {
            Write-Warning "�������󣬳������� $clonedRepoPathAtScriptDir..."
            Remove-Item -Path $clonedRepoPathAtScriptDir -Recurse -Force -ErrorAction SilentlyContinue
        }
        # �����׳��쳣��ȷ���ű����� $ErrorActionPreference = "Stop" ��ֹ
        throw
    }
}

Write-Host ("-" * 60)
Write-Host "�����������غ���������ɡ�"
