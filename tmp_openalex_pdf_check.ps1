$query = 'does physical workout effect mental health'
$encoded = [Uri]::EscapeDataString($query)
$url = "https://api.openalex.org/works?search=$encoded&per-page=20"
$response = (Invoke-WebRequest -Uri $url).Content | ConvertFrom-Json -AsHashtable -Depth 100
$items = @($response['results']) | Select-Object -First 8

$rows = foreach ($item in $items) {
    $title = [string]$item['title']
    $oaUrl = if ($item['open_access']) { [string]$item['open_access']['oa_url'] } else { '' }
    $bestPdf = if ($item['best_oa_location']) { [string]$item['best_oa_location']['pdf_url'] } else { '' }
    $primaryPdf = if ($item['primary_location']) { [string]$item['primary_location']['pdf_url'] } else { '' }

    $hasOa = -not [string]::IsNullOrWhiteSpace($oaUrl)
    $hasOtherPdf = (-not [string]::IsNullOrWhiteSpace($bestPdf)) -or (-not [string]::IsNullOrWhiteSpace($primaryPdf))
    $missedPdfIfOaOnly = (-not $hasOa) -and $hasOtherPdf

    [pscustomobject]@{
        title = $title
        'open_access.oa_url' = $oaUrl
        'best_oa_location.pdf_url' = $bestPdf
        'primary_location.pdf_url' = $primaryPdf
        'miss_pdf_if_oa_only' = $missedPdfIfOaOnly
    }
}

$rows |
    Select-Object title, 'open_access.oa_url', 'best_oa_location.pdf_url', 'primary_location.pdf_url', miss_pdf_if_oa_only |
    Format-Table -Wrap |
    Out-String -Width 4096 |
    Write-Output
