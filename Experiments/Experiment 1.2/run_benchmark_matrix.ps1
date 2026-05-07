param(
    [string]$Qwen3BModelPath = "C:\LLMs\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    [string]$Qwen7BModelPath = "C:\LLMs\models\Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    [string]$Mistral7BModelPath = "C:\LLMs\models\Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
    [int]$PatchGpuLayers = 35,
    [int]$ExplainerGpuLayers = 35,
    [string]$ImageName = "sound-forge-brain-sandbox:gpu",
    [switch]$RebuildImage
)

$ErrorActionPreference = "Stop"

$experiment12ResultDir = "Experiments/Experiment 1.2/results"
$experiment11ResultDir = "Experiments/Experiment 1.1/results"
$script:ConsumeRebuildImage = [bool]$RebuildImage

function Get-PythonCommand {
    foreach ($candidate in @(
        @{ Name = "py"; Args = @("-3") },
        @{ Name = "python"; Args = @() },
        @{ Name = "python3"; Args = @() }
    )) {
        if (Get-Command $candidate.Name -ErrorAction SilentlyContinue) {
            return $candidate
        }
    }
    return $null
}

function Get-RunFiles {
    param(
        [string]$Directory
    )

    if (-not (Test-Path $Directory)) {
        return @()
    }

    return @(Get-ChildItem $Directory -Filter "run_*.json" | ForEach-Object { $_.FullName })
}

function Get-NewRunFile {
    param(
        [string[]]$Before,
        [string]$Directory
    )

    $after = Get-RunFiles -Directory $Directory
    $newFiles = @($after | Where-Object { $_ -notin $Before })
    if ($newFiles.Count -ne 1) {
        throw "Expected exactly one new run artifact in $Directory, found $($newFiles.Count)."
    }
    return $newFiles[0]
}

function Invoke-WithOptionalRebuild {
    param(
        [string]$ScriptPath,
        [hashtable]$Arguments
    )

    if ($script:ConsumeRebuildImage) {
        $Arguments.RebuildImage = $true
        $script:ConsumeRebuildImage = $false
    }

    & $ScriptPath @Arguments | Out-Host
}

function Invoke-Experiment12Run {
    param(
        [string]$PatchModelPath,
        [string]$PatchContract,
        [string]$ExplanationMode,
        [string]$ExplanationRuntimeMode,
        [string]$PromptsFile,
        [string]$StatusFile,
        [string]$ExplainerModelPath = $null,
        [string]$FewShotFile = $null,
        [int]$FewShotCount = 0,
        [Nullable[int]]$PatchMaxTokens = $null,
        [int]$ExplainerMaxTokens = 96
    )

    $before = Get-RunFiles -Directory $experiment12ResultDir

    $arguments = @{
        PatchModelPath = $PatchModelPath
        PatchContract = $PatchContract
        ExplanationMode = $ExplanationMode
        ExplanationRuntimeMode = $ExplanationRuntimeMode
        PromptsFile = $PromptsFile
        StatusFile = $StatusFile
        ImageName = $ImageName
        PatchGpuLayers = $PatchGpuLayers
        ExplainerGpuLayers = $ExplainerGpuLayers
        ExplainerMaxTokens = $ExplainerMaxTokens
    }

    if ($PatchMaxTokens -ne $null) {
        $arguments.PatchMaxTokens = $PatchMaxTokens
    }

    if ($ExplainerModelPath) {
        $arguments.ExplainerModelPath = $ExplainerModelPath
    }
    if ($FewShotFile) {
        $arguments.FewShotFile = $FewShotFile
        $arguments.FewShotCount = $FewShotCount
    }

    Invoke-WithOptionalRebuild -ScriptPath ".\Experiments\Experiment 1.2\run_experiment_gpu.ps1" -Arguments $arguments
    return Get-NewRunFile -Before $before -Directory $experiment12ResultDir
}

function Invoke-Experiment11Run {
    param(
        [string]$ModelPath,
        [string]$PromptsFile,
        [string]$StatusFile,
        [int]$MaxTokens = 640
    )

    $before = Get-RunFiles -Directory $experiment11ResultDir

    $arguments = @{
        ModelPath = $ModelPath
        GpuLayers = $PatchGpuLayers
        PromptsFile = $PromptsFile
        StatusFile = $StatusFile
        ImageName = $ImageName
        MaxTokens = $MaxTokens
    }

    Invoke-WithOptionalRebuild -ScriptPath ".\Experiments\Experiment 1.1\run_experiment_gpu.ps1" -Arguments $arguments
    return Get-NewRunFile -Before $before -Directory $experiment11ResultDir
}

function Test-NeedsThreeShot {
    param(
        [string]$ResultFile
    )

    $resolvedResultFile = (Resolve-Path $ResultFile).Path
    $pythonCommand = Get-PythonCommand
    if ($null -ne $pythonCommand) {
        $pythonScript = @'
import json
import sys

same_value_ids = {"sparse_skip_same_shape_001", "sparse_same_value_noop_001"}
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)

needs_three_shot = False
for result in payload.get("results", []):
    if result.get("prompt_id") in same_value_ids:
        if not ((result.get("raw_sparse_fidelity") or {}).get("sparse_exact_match")):
            needs_three_shot = True
            break

sys.stdout.write("true" if needs_three_shot else "false")
'@
        $pythonArgs = @($pythonCommand.Args + @("-", $resolvedResultFile))
        $pythonOutput = $pythonScript | & $pythonCommand.Name @pythonArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to inspect $resolvedResultFile for C2 gating with Python."
        }
        return (($pythonOutput | Out-String).Trim() -eq "true")
    }

    if (Get-Command ConvertFrom-Json -ErrorAction SilentlyContinue) {
        $payload = [System.IO.File]::ReadAllText($resolvedResultFile) | ConvertFrom-Json
    } else {
        Write-Warning "Python and ConvertFrom-Json are unavailable on this host. Executing C2 three-shot variant conservatively."
        return $true
    }

    $sameValueIds = @("sparse_skip_same_shape_001", "sparse_same_value_noop_001")
    foreach ($result in $payload.results) {
        if ($sameValueIds -contains $result.prompt_id) {
            if (-not $result.raw_sparse_fidelity.sparse_exact_match) {
                return $true
            }
        }
    }
    return $false
}

function Invoke-SummaryAndReport {
    param(
        [string[]]$ResultFiles
    )

    $pythonCommand = Get-PythonCommand
    if ($null -eq $pythonCommand) {
        Write-Warning "Python launcher not found on the host. Skipping summarize_results.py and report generation."
        return
    }

    $summaryOutput = "Experiments/Experiment 1.2/logs/matrix_summary_latest.txt"
    $reportOutput = "Experiments/Experiment 1.2/reports/experiment-1-2-report.md"
    $reportDir = Split-Path -Path $reportOutput -Parent
    New-Item -ItemType Directory -Path $reportDir -Force | Out-Null

    $summaryArgs = @($pythonCommand.Args + @(
        ".\Experiments\Experiment 1.2\summarize_results.py"
    ) + $ResultFiles)
    & $pythonCommand.Name @summaryArgs | Tee-Object -FilePath $summaryOutput

    $reportArgs = @($pythonCommand.Args + @(
        ".\Experiments\Experiment 1.2\generate_experiment_1_2_report.py"
    ) + $ResultFiles + @("--output", $reportOutput))
    & $pythonCommand.Name @reportArgs

    Write-Host "Summary written to: $summaryOutput" -ForegroundColor DarkGray
    Write-Host "Report written to: $reportOutput" -ForegroundColor DarkGray
}

$testAFixture = "Experiments/Experiment 1.2/fixtures/test_a_latency_prompts.json"
$testA0Fixture = "Experiments/Experiment 1.1/fixtures/test_a_latency_prompts_explanation_first.json"
$testBFixture = "Experiments/Experiment 1.2/fixtures/test_b_numeric_pressure_prompts.json"
$testCFixture = "Experiments/Experiment 1.2/fixtures/test_c_sparse_ground_truth_prompts.json"
$fewShotSeed = "Experiments/Experiment 1.2/fixtures/fewshot_examples_seed.json"
$resultFiles = New-Object System.Collections.Generic.List[string]

Write-Host "Executing full Experiment 1.2 matrix" -ForegroundColor Cyan

$resultFiles.Add((Invoke-Experiment11Run `
    -ModelPath $Qwen7BModelPath `
    -PromptsFile $testA0Fixture `
    -StatusFile "Experiments/Experiment 1.1/logs/status_test_a0_qwen7b_single_pass.log"))

$resultFiles.Add((Invoke-Experiment12Run `
    -PatchModelPath $Qwen7BModelPath `
    -PatchContract "sparse_patch_only" `
    -ExplanationMode "off" `
    -ExplanationRuntimeMode "isolated" `
    -PromptsFile $testAFixture `
    -StatusFile "Experiments/Experiment 1.2/logs/status_test_a1_qwen7b_patch_only.log"))

$resultFiles.Add((Invoke-Experiment12Run `
    -PatchModelPath $Qwen7BModelPath `
    -PatchContract "sparse_patch_only" `
    -ExplanationMode "llm" `
    -ExplanationRuntimeMode "isolated" `
    -ExplainerModelPath $Qwen3BModelPath `
    -PromptsFile $testAFixture `
    -StatusFile "Experiments/Experiment 1.2/logs/status_test_a2_qwen7b_plus_qwen3b_explainer.log"))

$resultFiles.Add((Invoke-Experiment12Run `
    -PatchModelPath $Qwen7BModelPath `
    -PatchContract "compact_delta" `
    -ExplanationMode "off" `
    -ExplanationRuntimeMode "isolated" `
    -PatchMaxTokens 640 `
    -PromptsFile $testAFixture `
    -StatusFile "Experiments/Experiment 1.2/logs/status_test_a3_qwen7b_compact_delta.log"))

$resultFiles.Add((Invoke-Experiment12Run `
    -PatchModelPath $Qwen7BModelPath `
    -PatchContract "sparse_patch_only" `
    -ExplanationMode "off" `
    -ExplanationRuntimeMode "isolated" `
    -PromptsFile $testBFixture `
    -StatusFile "Experiments/Experiment 1.2/logs/status_test_b1_qwen7b_numeric.log"))

$resultFiles.Add((Invoke-Experiment12Run `
    -PatchModelPath $Mistral7BModelPath `
    -PatchContract "sparse_patch_only" `
    -ExplanationMode "off" `
    -ExplanationRuntimeMode "isolated" `
    -PromptsFile $testBFixture `
    -StatusFile "Experiments/Experiment 1.2/logs/status_test_b2_mistral7b_numeric.log"))

$resultFiles.Add((Invoke-Experiment12Run `
    -PatchModelPath $Qwen3BModelPath `
    -PatchContract "sparse_patch_only" `
    -ExplanationMode "off" `
    -ExplanationRuntimeMode "isolated" `
    -PromptsFile $testCFixture `
    -StatusFile "Experiments/Experiment 1.2/logs/status_test_c0_qwen3b_zero_shot.log"))

$c1Result = Invoke-Experiment12Run `
    -PatchModelPath $Qwen3BModelPath `
    -PatchContract "sparse_patch_only" `
    -ExplanationMode "off" `
    -ExplanationRuntimeMode "isolated" `
    -PromptsFile $testCFixture `
    -FewShotFile $fewShotSeed `
    -FewShotCount 2 `
    -StatusFile "Experiments/Experiment 1.2/logs/status_test_c1_qwen3b_two_shot.log"
$resultFiles.Add($c1Result)

if (Test-NeedsThreeShot -ResultFile $c1Result) {
    Write-Host "C1 still fails same-value sparse cases. Executing C2 three-shot variant." -ForegroundColor Yellow
    $resultFiles.Add((Invoke-Experiment12Run `
        -PatchModelPath $Qwen3BModelPath `
        -PatchContract "sparse_patch_only" `
        -ExplanationMode "off" `
        -ExplanationRuntimeMode "isolated" `
        -PromptsFile $testCFixture `
        -FewShotFile $fewShotSeed `
        -FewShotCount 3 `
        -StatusFile "Experiments/Experiment 1.2/logs/status_test_c2_qwen3b_three_shot.log"))
} else {
    Write-Host "C1 passed the same-value sparse cases. Skipping C2 three-shot variant." -ForegroundColor Green
}

Invoke-SummaryAndReport -ResultFiles $resultFiles.ToArray()
