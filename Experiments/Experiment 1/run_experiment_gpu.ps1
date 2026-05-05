param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,

    [int]$GpuLayers = 35,
    [Nullable[int]]$Limit = $null,
    [string[]]$PromptId = @(),
    [string]$StatusFile = "Experiments/Experiment 1/logs/status_latest.log",
    [string]$ImageName = "sound-forge-brain-sandbox:gpu",
    [switch]$RebuildImage,
    [string]$ChatFormat,
    [string]$PromptsFile,
    [int]$NCtx = 4096,
    [int]$MaxTokens = 96,
    [double]$Temperature = 0.0,
    [double]$TopP = 1.0,
    [double]$RepeatPenalty = 1.0,
    [int]$Seed = 42,
    [Nullable[int]]$NThreads = $null
)

$ErrorActionPreference = "Stop"

$experimentDir = (Resolve-Path $PSScriptRoot).Path
$repoRoot = (Resolve-Path (Join-Path $experimentDir "..\..")).Path

$resolvedModelPath = (Resolve-Path $ModelPath).Path
$modelDirectory = Split-Path -Path $resolvedModelPath -Parent
$modelFileName = Split-Path -Path $resolvedModelPath -Leaf
$dockerfilePath = Join-Path $experimentDir "Dockerfile.gpu"

if (-not (Test-Path $dockerfilePath)) {
    throw "Dockerfile not found: $dockerfilePath"
}

$imageExists = $true
try {
    docker image inspect $ImageName *> $null
} catch {
    $imageExists = $false
}

if ($RebuildImage -or -not $imageExists) {
    Write-Host "Building GPU image: $ImageName" -ForegroundColor Cyan

    $buildArgs = @(
        "build",
        "--progress=plain",
        "-f",
        $dockerfilePath,
        "-t",
        $ImageName
    )

    foreach ($proxyName in @("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy")) {
        $proxyValue = [Environment]::GetEnvironmentVariable($proxyName)
        if (-not [string]::IsNullOrWhiteSpace($proxyValue)) {
            $buildArgs += @("--build-arg", "${proxyName}=$proxyValue")
        }
    }

    $buildArgs += $experimentDir

    & docker @buildArgs

    if ($LASTEXITCODE -ne 0) {
        throw "Docker image build failed with code $LASTEXITCODE"
    }
}

$workspaceMount = "type=bind,source=$repoRoot,target=/workspace"
$modelMount = "type=bind,source=$modelDirectory,target=/models,readonly"
$cacheVolume = "type=volume,source=brain-sandbox-cache,target=/root/.cache"

$containerArgs = @(
    "run",
    "--rm",
    "--gpus",
    "all",
    "--mount",
    $workspaceMount,
    "--mount",
    $modelMount,
    "--mount",
    $cacheVolume,
    "-w",
    "/workspace",
    $ImageName,
    "python3",
    "Experiments/Experiment 1/brain_sandbox.py",
    "--model-path",
    "/models/$modelFileName",
    "--n-gpu-layers",
    $GpuLayers,
    "--status-file",
    $StatusFile,
    "--n-ctx",
    $NCtx,
    "--max-tokens",
    $MaxTokens,
    "--temperature",
    $Temperature,
    "--top-p",
    $TopP,
    "--repeat-penalty",
    $RepeatPenalty,
    "--seed",
    $Seed
)

if ($Limit -ne $null) {
    $containerArgs += @("--limit", $Limit)
}

if ($ChatFormat) {
    $containerArgs += @("--chat-format", $ChatFormat)
}

if ($NThreads -ne $null) {
    $containerArgs += @("--n-threads", $NThreads)
}

if ($PromptsFile) {
    $containerArgs += @("--prompts-file", $PromptsFile)
}

foreach ($id in $PromptId) {
    $containerArgs += @("--prompt-id", $id)
}

Write-Host "Running Experiment 1 against model: $resolvedModelPath" -ForegroundColor Green
Write-Host "Workspace mount: $repoRoot -> /workspace" -ForegroundColor DarkGray
Write-Host "Model mount: $modelDirectory -> /models (read-only)" -ForegroundColor DarkGray
Write-Host "Status log will be written to: $StatusFile" -ForegroundColor DarkGray

& docker @containerArgs

if ($LASTEXITCODE -ne 0) {
    throw "Experiment container exited with code $LASTEXITCODE"
}
