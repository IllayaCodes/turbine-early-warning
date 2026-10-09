docker rm -f tew 2>$null | Out-Null
docker run -d --rm --name tew -p 8000:8000 turbine-early-warning | Out-Null
Write-Host "waiting for the server..."
$ok = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        Invoke-RestMethod http://127.0.0.1:8000/health | Out-Null
        $ok = $true
        break
    } catch {
        Start-Sleep -Seconds 1
    }
}
if (-not $ok) {
    docker logs tew
    docker rm -f tew | Out-Null
    throw "the server did not answer"
}
"--- /health"
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 5
"--- /alerts?turbine=6"
Invoke-RestMethod "http://127.0.0.1:8000/alerts?turbine=6" | ConvertTo-Json -Depth 5
"--- /predict with a real row"
if (Test-Path make_demo.py) {
    $out = python make_demo.py
    $out | Select-Object -First 1
    $body = $out | Select-Object -Last 1
    Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 5
} else {
    "make_demo.py not found, skipping"
}
"--- waiting for the container health check (about 40 seconds)"
Start-Sleep -Seconds 40
docker ps --filter name=tew --format "{{.Names}}  {{.Status}}"
docker stop tew | Out-Null
"container stopped"