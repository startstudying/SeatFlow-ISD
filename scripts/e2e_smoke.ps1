param(
    [string]$ComposeFile = "deployment/docker-compose.yml",
    [string]$EnvFile = "deployment/.env",
    [string]$FastApiBaseUrl = "http://127.0.0.1:8000",
    [string]$JavaBaseUrl = "http://127.0.0.1:8080",
    [string]$FrontendBaseUrl = "http://127.0.0.1:3000",
    [string]$MlflowBaseUrl = "http://127.0.0.1:5000",
    [string]$ModelId = "missing-checkpoint",
    [string]$Provider = "sample",
    [string]$TradeDate = "2026-01-15",
    [string]$StockId = "600000.SH",
    [switch]$SkipComposeConfig
)

$ErrorActionPreference = "Stop"

function Invoke-Json {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [string]$Method = "GET",
        [object]$Body = $null
    )

    $params = @{
        Uri = $Uri
        Method = $Method
        Headers = @{ Accept = "application/json" }
        TimeoutSec = 10
    }
    if ($null -ne $Body) {
        $params.ContentType = "application/json"
        $params.Body = ($Body | ConvertTo-Json -Depth 20)
    }
    Invoke-RestMethod @params
}

function Test-HttpStatus {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [int[]]$AllowedStatus = @(200)
    )

    try {
        $response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 10
        if ($AllowedStatus -notcontains [int]$response.StatusCode) {
            throw "Unexpected status $($response.StatusCode) for $Uri"
        }
        return [int]$response.StatusCode
    } catch {
        $response = $_.Exception.Response
        if ($null -eq $response -or $null -eq $response.StatusCode) {
            throw
        }
        $status = [int]$response.StatusCode
        if ($AllowedStatus -contains [int]$status) {
            return [int]$status
        }
        throw
    }
}

function Invoke-ExpectHttpError {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][object]$Body,
        [int[]]$AllowedStatus = @(404, 409)
    )

    try {
        Invoke-Json -Uri $Uri -Method POST -Body $Body | Out-Null
        throw "Request unexpectedly succeeded: $Uri"
    } catch {
        $response = $_.Exception.Response
        if ($null -eq $response -or $null -eq $response.StatusCode) {
            throw
        }
        $status = [int]$response.StatusCode
        if ($AllowedStatus -notcontains $status) {
            throw
        }
        return $status
    }
}

if (-not (Test-Path $EnvFile)) {
    $example = "deployment/.env.example"
    if (Test-Path $example) {
        $EnvFile = $example
    }
}

if (-not $SkipComposeConfig) {
    docker compose --env-file $EnvFile -f $ComposeFile config --quiet
}

$fastHealth = Invoke-Json "$FastApiBaseUrl/health"
if ($fastHealth.status -ne "ok") {
    throw "FastAPI health is not ok: $($fastHealth | ConvertTo-Json -Compress)"
}

$javaHealth = Invoke-Json "$JavaBaseUrl/ready"
if ($javaHealth.data.status -ne "UP") {
    throw "Java readiness is not UP: $($javaHealth | ConvertTo-Json -Compress)"
}

$upstream = Invoke-Json "$JavaBaseUrl/api/v1/upstream/fastapi/health"
if ($upstream.data.status -ne "ok") {
    throw "Java cannot reach FastAPI health: $($upstream | ConvertTo-Json -Compress)"
}

$mlflowStatus = Test-HttpStatus "$MlflowBaseUrl/health" @(200)

$predictionBody = @{
    trade_date = $TradeDate
    target_stock_id = $StockId
}
$predictionUri = "$FastApiBaseUrl/api/v1/providers/$Provider/models/$ModelId/predict"
$predictionStatus = Invoke-ExpectHttpError -Uri $predictionUri -Body $predictionBody
$javaPredictionUri = "$JavaBaseUrl/api/v1/providers/$Provider/models/$ModelId/predict"
$javaPredictionStatus = Invoke-ExpectHttpError -Uri $javaPredictionUri -Body $predictionBody

$frontendStatus = Test-HttpStatus "$FrontendBaseUrl/" @(200)

[pscustomobject]@{
    ok = $true
    fastapi_health = $fastHealth
    java_readiness = $javaHealth.data
    upstream_fastapi = $upstream.data
    mlflow_status = $mlflowStatus
    fastapi_prediction_fail_closed_status = $predictionStatus
    java_prediction_fail_closed_status = $javaPredictionStatus
    frontend_status = $frontendStatus
} | ConvertTo-Json -Depth 20
