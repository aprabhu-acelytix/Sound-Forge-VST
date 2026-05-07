param(
    [Parameter(Mandatory = $true)]
    [string]$PatchModelPath,

    [string]$ExplainerModelPath,
    [ValidateSet("sparse_patch_only", "compact_delta")]
    [string]$PatchContract = "sparse_patch_only",
    [ValidateSet("off", "deterministic", "llm")]
    [string]$ExplanationMode = "off",
    [ValidateSet("isolated", "serialized")]
    [string]$ExplanationRuntimeMode = "isolated",
    [Nullable[int]]$Limit = $null,
    [string[]]$PromptId = @(),
    [string]$PromptsFile,
    [string]$FewShotFile,
    [int]$FewShotCount = 0,
    [string]$StatusFile = "Experiments/Experiment 1.2/logs/status_latest.log",
    [string]$ImageName = "sound-forge-brain-sandbox:gpu",
    [switch]$RebuildImage,
    [string]$PatchChatFormat,
    [string]$ExplainerChatFormat,
    [int]$PatchNCtx = 4096,
    [int]$ExplainerNCtx = 2048,
    [int]$PatchMaxTokens = 384,
    [int]$ExplainerMaxTokens = 96,
    [double]$PatchTemperature = 0.0,
    [double]$ExplainerTemperature = 0.0,
    [double]$PatchTopP = 1.0,
    [double]$ExplainerTopP = 1.0,
    [double]$PatchRepeatPenalty = 1.0,
    [double]$ExplainerRepeatPenalty = 1.0,
    [int]$PatchGpuLayers = 35,
    [int]$ExplainerGpuLayers = 35,
    [Nullable[int]]$PatchNThreads = $null,
    [Nullable[int]]$ExplainerNThreads = $null,
    [int]$Seed = 42,
    [int]$WorkerQueueSize = 64
)

$ErrorActionPreference = "Stop"

$experimentDir = (Resolve-Path $PSScriptRoot).Path
$repoRoot = (Resolve-Path (Join-Path $experimentDir "..\..")).Path
$dockerfilePath = Join-Path $repoRoot "Experiments\Experiment 1\Dockerfile.gpu"

try {
    docker version *> $null
} catch {
    throw "Docker daemon is not reachable. Start Docker Desktop and verify the Linux engine is running before retrying."
}

$resolvedPatchModelPath = (Resolve-Path $PatchModelPath).Path
$patchModelDirectory = Split-Path -Path $resolvedPatchModelPath -Parent
$patchModelFileName = Split-Path -Path $resolvedPatchModelPath -Leaf

if (-not $PSBoundParameters.ContainsKey('PatchMaxTokens')) {
    if ($PatchContract -eq 'sparse_patch_only') {
        $PatchMaxTokens = 640
    } elseif ($PatchContract -eq 'compact_delta') {
        $PatchMaxTokens = 256
    }
}

$resolvedExplainerModelPath = $null
$explainerModelDirectory = $null
$explainerModelFileName = $null
if ($ExplainerModelPath) {
    $resolvedExplainerModelPath = (Resolve-Path $ExplainerModelPath).Path
    $explainerModelDirectory = Split-Path -Path $resolvedExplainerModelPath -Parent
    $explainerModelFileName = Split-Path -Path $resolvedExplainerModelPath -Leaf
}

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
    Write-Host "Building shared GPU image: $ImageName" -ForegroundColor Cyan

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

    $buildArgs += (Join-Path $repoRoot "Experiments\Experiment 1")

    & docker @buildArgs

    if ($LASTEXITCODE -ne 0) {
        throw "Docker image build failed with code $LASTEXITCODE"
    }
}

$workspaceMount = "type=bind,source=$repoRoot,target=/workspace"
$cacheVolume = "type=volume,source=brain-sandbox-cache,target=/root/.cache"
$mounts = @(
    "--mount", $workspaceMount,
    "--mount", $cacheVolume
)

$mountedDirectories = @{}
$mountedDirectories[$patchModelDirectory] = "/models_patch"
if ($resolvedExplainerModelPath) {
    if (-not $mountedDirectories.ContainsKey($explainerModelDirectory)) {
        $mountedDirectories[$explainerModelDirectory] = "/models_explainer"
    }
}

foreach ($kvp in $mountedDirectories.GetEnumerator()) {
    $mounts += @("--mount", "type=bind,source=$($kvp.Key),target=$($kvp.Value),readonly")
}

$containerArgs = @(
    "run",
    "--rm",
    "--gpus",
    "all"
)
$containerArgs += $mounts
$containerArgs += @(
    "-w",
    "/workspace",
    $ImageName,
    "python3",
    "Experiments/Experiment 1.2/brain_sandbox_experiment_1_2.py",
    "--patch-model-path",
    "$($mountedDirectories[$patchModelDirectory])/$patchModelFileName",
    "--patch-contract",
    $PatchContract,
    "--explanation-mode",
    $ExplanationMode,
    "--explanation-runtime-mode",
    $ExplanationRuntimeMode,
    "--patch-n-gpu-layers",
    $PatchGpuLayers,
    "--explainer-n-gpu-layers",
    $ExplainerGpuLayers,
    "--status-file",
    $StatusFile,
    "--patch-n-ctx",
    $PatchNCtx,
    "--explainer-n-ctx",
    $ExplainerNCtx,
    "--patch-max-tokens",
    $PatchMaxTokens,
    "--explainer-max-tokens",
    $ExplainerMaxTokens,
    "--patch-temperature",
    $PatchTemperature,
    "--explainer-temperature",
    $ExplainerTemperature,
    "--patch-top-p",
    $PatchTopP,
    "--explainer-top-p",
    $ExplainerTopP,
    "--patch-repeat-penalty",
    $PatchRepeatPenalty,
    "--explainer-repeat-penalty",
    $ExplainerRepeatPenalty,
    "--seed",
    $Seed,
    "--worker-queue-size",
    $WorkerQueueSize
)

if ($resolvedExplainerModelPath) {
    $containerArgs += @(
        "--explainer-model-path",
        "$($mountedDirectories[$explainerModelDirectory])/$explainerModelFileName"
    )
}

if ($Limit -ne $null) {
    $containerArgs += @("--limit", $Limit)
}

if ($PatchChatFormat) {
    $containerArgs += @("--patch-chat-format", $PatchChatFormat)
}

if ($ExplainerChatFormat) {
    $containerArgs += @("--explainer-chat-format", $ExplainerChatFormat)
}

if ($PatchNThreads -ne $null) {
    $containerArgs += @("--patch-n-threads", $PatchNThreads)
}

if ($ExplainerNThreads -ne $null) {
    $containerArgs += @("--explainer-n-threads", $ExplainerNThreads)
}

if ($PromptsFile) {
    $containerArgs += @("--prompts-file", $PromptsFile)
}

if ($FewShotFile) {
    $containerArgs += @("--few-shot-file", $FewShotFile)
}

if ($FewShotCount -gt 0) {
    $containerArgs += @("--few-shot-count", $FewShotCount)
}

foreach ($id in $PromptId) {
    $containerArgs += @("--prompt-id", $id)
}

Write-Host "Running Experiment 1.2 against patch model: $resolvedPatchModelPath" -ForegroundColor Green
Write-Host "Patch contract: $PatchContract" -ForegroundColor DarkGray
Write-Host "Explanation mode: $ExplanationMode / $ExplanationRuntimeMode" -ForegroundColor DarkGray
if ($resolvedExplainerModelPath) {
    Write-Host "Explainer model: $resolvedExplainerModelPath" -ForegroundColor DarkGray
}
Write-Host "Workspace mount: $repoRoot -> /workspace" -ForegroundColor DarkGray
Write-Host "Status log will be written to: $StatusFile" -ForegroundColor DarkGray

& docker @containerArgs

if ($LASTEXITCODE -ne 0) {
    throw "Experiment 1.2 container exited with code $LASTEXITCODE"
}
