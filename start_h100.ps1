$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$runtime = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $runtime)) {
    $basePython = 'python'
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $detected = & py -3.13 -c 'import sys; print(sys.executable)' 2>$null
        if ($LASTEXITCODE -eq 0) { $basePython = "$detected".Trim() }
    }
    & (Join-Path $PSScriptRoot 'setup.ps1') -Device cuda -Python $basePython
}
if (-not (Test-Path -LiteralPath $runtime)) { throw 'Environment setup did not complete' }
& $runtime (Join-Path $PSScriptRoot 'runner.py') check --cnn-pool first --models 'LSTM,CNN-LSTM,MS-AgentNet,Transformer,CNN-Transformer' --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed; do not start the search' }
& $runtime -u (Join-Path $PSScriptRoot 'start_h100.py') --gpu-index 0
if ($LASTEXITCODE -ne 0) { throw 'H100 launcher stopped; inspect logs' }
