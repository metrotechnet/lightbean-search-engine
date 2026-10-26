$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Get-Command python -ErrorAction SilentlyContinue

if (-not $python) {
    throw "Python is not installed or not on PATH."
}

$envPath = Join-Path $scriptDir '.env'
if (-not (Test-Path $envPath)) {
    throw "Missing .env file at $envPath"
}

Get-Content $envPath | ForEach-Object {
    if ($_ -match '^([^#=]+)=(.*)$') {
        $name = $matches[1].Trim()
        $value = $matches[2].Trim()
        if ($name -and $value) {
            [System.Environment]::SetEnvironmentVariable($name, $value, 'Process')
        }
    }
}

& python (Join-Path $scriptDir 'scripts\chat_meilisearch_questions.py')

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
