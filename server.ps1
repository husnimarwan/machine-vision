<#
  Kelola server web MachineVision (start / stop / status).

  Biasanya dipanggil lewat start.bat atau stop.bat, tapi bisa juga manual:
      powershell -ExecutionPolicy Bypass -File .\server.ps1 start

  Catatan: server dijalankan dengan jendela terpisah (minimized) supaya log
  tetap bisa dilihat, dan start.bat tidak ikut menutup server saat selesai.
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("start", "stop", "status")]
    [string]$Action
)

$root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$port   = 5000
$url    = "http://127.0.0.1:$port/"

function Test-ServerUp {
    try {
        $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
        return ($r.StatusCode -eq 200)
    } catch {
        return $false
    }
}

function Get-ServerProcesses {
    $found = @()

    # 1) Proses python milik venv proyek ini: server + parent auto-reloader.
    if (Test-Path $python) {
        $exe = (Resolve-Path $python).Path
        $found += @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Where-Object { $_.ExecutablePath -and ($_.ExecutablePath -ieq $exe) })
    }

    # 2) Siapa pun yang memegang port 5000 (mis. dijalankan pakai python lain),
    #    berikut parent-nya kalau parent itu juga python (auto-reloader).
    foreach ($conn in @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)) {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($conn.OwningProcess)" -ErrorAction SilentlyContinue
        if ($proc) {
            $found += $proc
            $parent = Get-CimInstance Win32_Process -Filter "ProcessId=$($proc.ParentProcessId)" -ErrorAction SilentlyContinue
            if ($parent -and $parent.Name -eq "python.exe") { $found += $parent }
        }
    }

    return @($found | Sort-Object ProcessId -Unique)
}

function Stop-Server {
    $procs = Get-ServerProcesses
    if ($procs.Count -eq 0) {
        Write-Host "  Server tidak sedang berjalan."
        return
    }

    foreach ($p in $procs) {
        Write-Host ("  menghentikan PID " + $p.ProcessId + " (" + $p.Name + ")")
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }

    # Tunggu port 5000 benar-benar lepas sebelum keluar.
    for ($i = 0; $i -lt 20; $i++) {
        if (-not (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)) { break }
        Start-Sleep -Milliseconds 300
    }
}

switch ($Action) {
    "start" {
        if (Test-ServerUp) {
            Write-Host "Server sudah berjalan: $url"
            Start-Process $url
            exit 0
        }

        if (-not (Test-Path $python)) {
            Write-Host "[ERROR] Python venv tidak ditemukan:"
            Write-Host "        $python"
            Write-Host "Buat dulu: python -m venv .venv  lalu install requirements.txt"
            exit 1
        }

        Write-Host "Menjalankan server MachineVision..."

        # Jendela terpisah, minimized, judulnya "MachineVision Server".
        # Tutup jendela itu = server berhenti (atau pakai stop.bat).
        Start-Process -FilePath "cmd.exe" `
            -ArgumentList "/c", "start", '"MachineVision Server"', "/min", "`"$python`"", "app.py" `
            -WorkingDirectory $root | Out-Null

        Write-Host "Menunggu server siap (memuat model YOLO)..."
        $up = $false
        for ($i = 0; $i -lt 120; $i++) {
            if (Test-ServerUp) { $up = $true; break }
            Start-Sleep -Milliseconds 500
        }

        if (-not $up) {
            Write-Host "[ERROR] Server tidak merespons."
            Write-Host "        Cek jendela 'MachineVision Server' untuk melihat pesan errornya."
            exit 1
        }

        Write-Host "Server siap: $url"
        Start-Process $url
        exit 0
    }

    "stop" {
        Stop-Server
        Write-Host "Server dihentikan."
        exit 0
    }

    "status" {
        if (Test-ServerUp) {
            Write-Host "Server JALAN: $url"
        } else {
            Write-Host "Server TIDAK jalan."
        }
        exit 0
    }
}
