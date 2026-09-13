param(
    [ValidateSet('cpu','cuda')][string]$Device = 'cuda',
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
$envDirectory = Join-Path $PSScriptRoot '.venv'
if (Test-Path -LiteralPath $envDirectory) { throw '.venv already exists; inspect/reuse it rather than overwrite.' }
& $Python -c "import sys; assert sys.version_info[:2] == (3,13), 'Please use Python 3.13 to match the verified environment'"
if ($LASTEXITCODE -ne 0) { throw 'Python version check failed' }
& $Python -m venv $envDirectory
if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
$runtime = Join-Path $envDirectory 'Scripts\python.exe'
$index = if ($Device -eq 'cuda') { 'https://download.pytorch.org/whl/cu128' } else { 'https://download.pytorch.org/whl/cpu' }
& $runtime -m pip install --no-cache-dir 'torch==2.11.0' --index-url $index
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $runtime -m pip install --no-cache-dir -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $runtime (Join-Path $PSScriptRoot 'runner.py') check --cnn-pool first --models 'LSTM,CNN-LSTM,MS-AgentNet,Transformer,CNN-Transformer' --device $Device
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed; do not start formal search' }
Write-Output "Ready. Use run_worker.ps1 with -Python `"$runtime`" and a uniquely assigned shard index."
