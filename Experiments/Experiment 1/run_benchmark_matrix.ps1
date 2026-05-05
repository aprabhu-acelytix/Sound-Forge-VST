param(
    [string]$Qwen3BModelPath = "C:\LLMs\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    [string]$Qwen7BModelPath = "C:\LLMs\models\Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    [int]$GpuLayers = 35,
    [switch]$IncludeAdversarial,
    [switch]$RebuildImage,
    [string]$ImageName = "sound-forge-brain-sandbox:gpu"
)

$ErrorActionPreference = "Stop"

function Invoke-BrainRun {
    param(
        [string]$ModelPath,
        [string]$PromptsFile,
        [string]$StatusFile,
        [switch]$Rebuild
    )

    $arguments = @{
        ModelPath = $ModelPath
        GpuLayers = $GpuLayers
        StatusFile = $StatusFile
        PromptsFile = $PromptsFile
        ImageName = $ImageName
    }

    if ($Rebuild) {
        $arguments["RebuildImage"] = $true
    }

    & ".\Experiments\Experiment 1\run_experiment_gpu.ps1" @arguments
}

Invoke-BrainRun `
    -ModelPath $Qwen3BModelPath `
    -PromptsFile "Experiments/Experiment 1/fixtures/test_prompts.json" `
    -StatusFile "Experiments/Experiment 1/logs/status_qwen3b_baseline.log" `
    -Rebuild:$RebuildImage

Invoke-BrainRun `
    -ModelPath $Qwen7BModelPath `
    -PromptsFile "Experiments/Experiment 1/fixtures/test_prompts.json" `
    -StatusFile "Experiments/Experiment 1/logs/status_qwen7b_baseline.log"

if ($IncludeAdversarial) {
    Invoke-BrainRun `
        -ModelPath $Qwen3BModelPath `
        -PromptsFile "Experiments/Experiment 1/fixtures/adversarial_prompts.json" `
        -StatusFile "Experiments/Experiment 1/logs/status_qwen3b_adversarial.log"

    Invoke-BrainRun `
        -ModelPath $Qwen7BModelPath `
        -PromptsFile "Experiments/Experiment 1/fixtures/adversarial_prompts.json" `
        -StatusFile "Experiments/Experiment 1/logs/status_qwen7b_adversarial.log"
}
