$query = 'does physical workout effect mental health'
$encoded = [Uri]::EscapeDataString($query)

$openAlexUrl = "https://api.openalex.org/works?search=$encoded&per-page=10"
$openAlex = (Invoke-WebRequest -Uri $openAlexUrl).Content | ConvertFrom-Json -AsHashtable -Depth 100
$openAlexItems = @($openAlex['results'])
$openAlexPdfCount = 0
"=== OpenAlex (limit 10) ==="
$idx = 0
foreach ($item in $openAlexItems) {
    $idx++
    $title = if ($item['title']) { [string]$item['title'] } else { [string]$item['display_name'] }
    $doi = [string]$item['doi']
    $landingUrl = if ($item['primary_location']) { [string]$item['primary_location']['landing_page_url'] } else { '' }
    $pdfCandidates = @(
        $(if ($item['open_access']) { $item['open_access']['oa_url'] }),
        $(if ($item['best_oa_location']) { $item['best_oa_location']['pdf_url'] }),
        $(if ($item['primary_location']) { $item['primary_location']['pdf_url'] })
    ) | Where-Object { $_ -and -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ } | Select-Object -Unique

    if ($pdfCandidates.Count -gt 0) { $openAlexPdfCount++ }

    "[$idx] Title: $title"
    "    DOI: $doi"
    "    Landing URL: $landingUrl"
    "    PDF-like URLs: " + $(if ($pdfCandidates.Count -gt 0) { $pdfCandidates -join ' | ' } else { '(none)' })
}

$semanticScholarUrl = "https://api.semanticscholar.org/graph/v1/paper/search/bulk?query=$encoded&fields=title,externalIds,url,openAccessPdf"
$semanticScholar = (Invoke-WebRequest -Uri $semanticScholarUrl).Content | ConvertFrom-Json -AsHashtable -Depth 20
$semanticScholarItems = @($semanticScholar['data']) | Select-Object -First 10
$semanticScholarPdfCount = 0
"=== Semantic Scholar (limit 10) ==="
$idx = 0
foreach ($item in $semanticScholarItems) {
    $idx++
    $title = [string]$item['title']
    $doi = if ($item['externalIds']) { [string]$item['externalIds']['DOI'] } else { '' }
    $landingUrl = [string]$item['url']
    $pdfCandidates = @(
        $(if ($item['openAccessPdf']) { $item['openAccessPdf']['url'] })
    ) | Where-Object { $_ -and -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ } | Select-Object -Unique

    if ($pdfCandidates.Count -gt 0) { $semanticScholarPdfCount++ }

    "[$idx] Title: $title"
    "    DOI: $doi"
    "    Landing URL: $landingUrl"
    "    PDF-like URLs: " + $(if ($pdfCandidates.Count -gt 0) { $pdfCandidates -join ' | ' } else { '(none)' })
}

"=== Summary Counts ==="
"OpenAlex items with >=1 non-empty PDF URL candidate: $openAlexPdfCount / $($openAlexItems.Count)"
"Semantic Scholar items with >=1 non-empty PDF URL candidate: $semanticScholarPdfCount / $($semanticScholarItems.Count)"
