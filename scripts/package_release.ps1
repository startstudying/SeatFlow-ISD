param(
    [string]$StatusFile = "artifacts/status/phase-state.json",
    [Parameter(Mandatory = $true)][string]$BlindManifest,
    [Parameter(Mandatory = $true)][string]$CompetitionContract,
    [Parameter(Mandatory = $true)][string]$SubmissionFile,
    [string]$OutputDir = "artifacts/release"
)

$ErrorActionPreference = "Stop"

function Fail([string]$Message) {
    throw "SeatFlow release gate failed: $Message"
}

function Resolve-ExistingFile([string]$PathValue, [string]$Name) {
    $item = Get-Item -LiteralPath $PathValue -ErrorAction SilentlyContinue
    if ($null -eq $item -or $item.PSIsContainer) {
        Fail "$Name file does not exist: $PathValue"
    }
    return $item.FullName
}

function Get-FileSha256([string]$PathValue) {
    return (Get-FileHash -LiteralPath $PathValue -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-TrueFlag($Status, [string]$Name) {
    if ($Status.PSObject.Properties.Name -notcontains $Name -or $Status.$Name -ne $true) {
        Fail "$Name must be true"
    }
}

function Assert-EvidenceHash($Hashes, [string[]]$CandidateKeys, [string]$ActualHash, [string]$Label) {
    $hashKeys = @()
    if ($null -ne $Hashes) {
        $hashKeys = @($Hashes.PSObject.Properties | Where-Object { $_.MemberType -eq "NoteProperty" })
    }
    if ($null -eq $Hashes -or $hashKeys.Count -eq 0) {
        Fail "evidence_hashes must be a non-empty map"
    }
    foreach ($key in $CandidateKeys) {
        if ($Hashes.PSObject.Properties.Name -contains $key) {
            $expected = [string]$Hashes.$key
            if ($expected -notmatch '^[A-Fa-f0-9]{64}$') {
                Fail "$Label hash for evidence_hashes.$key must be a SHA256 hex string"
            }
            if ($expected.ToLowerInvariant() -ne $ActualHash) {
                Fail "$Label hash mismatch for evidence_hashes.$key"
            }
            return
        }
    }
    Fail "$Label hash is required in evidence_hashes"
}

function Assert-BlindRecords($Blind, [string]$Key) {
    if ($Blind.PSObject.Properties.Name -notcontains $Key -or $Blind.$Key -isnot [array]) {
        Fail "blind manifest must include non-empty $Key records"
    }
    if ($Blind.$Key.Count -eq 0) {
        Fail "blind manifest must include non-empty $Key records"
    }
    foreach ($record in $Blind.$Key) {
        if ($record.PSObject.Properties.Name -notcontains "path") {
            Fail "blind manifest $Key records require path"
        }
        if ([string]::IsNullOrWhiteSpace([string]$record.path)) {
            Fail "blind manifest $Key records require path"
        }
        if ($record.PSObject.Properties.Name -notcontains "sha256") {
            Fail "blind manifest $Key records require sha256"
        }
        $recordHash = [string]$record.sha256
        if ($recordHash -notmatch '^[A-Fa-f0-9]{64}$') {
            Fail "blind manifest $Key records require valid SHA256"
        }
    }
}

$statusPath = Resolve-ExistingFile $StatusFile "phase-state"
$blindPath = Resolve-ExistingFile $BlindManifest "blind manifest"
$contractPath = Resolve-ExistingFile $CompetitionContract "competition contract"
$submissionPath = Resolve-ExistingFile $SubmissionFile "submission"

try {
    $status = Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json
} catch {
    Fail "phase-state must be readable JSON"
}

foreach ($flag in @(
    "engineering_complete",
    "data_ready",
    "model_valid",
    "backtest_valid",
    "competition_ready"
)) {
    Assert-TrueFlag $status $flag
}

if ($status.data_scope -ne "real") {
    Fail "data_scope must be real"
}

$seatScope = $status.seat_detail_scope
if ([string]::IsNullOrWhiteSpace($seatScope)) {
    $seatScope = $status.scope
}
if ($seatScope -ne "complete_seat") {
    Fail "seat_detail_scope must be complete_seat"
}

try {
    $blind = Get-Content -LiteralPath $blindPath -Raw -Encoding UTF8 | ConvertFrom-Json
} catch {
    Fail "blind manifest must be readable JSON"
}
if ($blind.blind -ne $true) {
    Fail "blind manifest must set blind=true"
}
if ($blind.label_feedback_forbidden -ne $true) {
    Fail "blind manifest must set label_feedback_forbidden=true"
}
Assert-BlindRecords $blind "features"
Assert-BlindRecords $blind "labels"

$statusHash = Get-FileSha256 $statusPath
$blindHash = Get-FileSha256 $blindPath
$contractHash = Get-FileSha256 $contractPath
$submissionHash = Get-FileSha256 $submissionPath

Assert-EvidenceHash $status.evidence_hashes @("blind_manifest", "blind_manifest_sha256") $blindHash "blind manifest"
Assert-EvidenceHash $status.evidence_hashes @("competition_contract", "competition_contract_sha256") $contractHash "competition contract"
Assert-EvidenceHash $status.evidence_hashes @("submission", "submission_sha256") $submissionHash "submission"

$outputPath = Join-Path $OutputDir "release-manifest.json"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$resolvedOutputPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($outputPath)

$manifest = [ordered]@{
    generated_at = (Get-Date).ToUniversalTime().ToString("o")
    release_gate = "passed"
    phase_state = [ordered]@{
        path = $statusPath
        sha256 = $statusHash
        engineering_complete = $status.engineering_complete
        data_ready = $status.data_ready
        model_valid = $status.model_valid
        backtest_valid = $status.backtest_valid
        competition_ready = $status.competition_ready
        data_scope = $status.data_scope
        seat_detail_scope = $seatScope
    }
    evidence = [ordered]@{
        blind_manifest = [ordered]@{ path = $blindPath; sha256 = $blindHash }
        competition_contract = [ordered]@{ path = $contractPath; sha256 = $contractHash }
        submission = [ordered]@{ path = $submissionPath; sha256 = $submissionHash }
    }
    output_boundary = "manifest_only_no_raw_data_no_secrets_no_checkpoint_contents"
}

$manifestJson = $manifest | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText(
    $resolvedOutputPath,
    $manifestJson,
    [System.Text.UTF8Encoding]::new($false)
)
Write-Output "SeatFlow release manifest generated: $outputPath"
