# ════════════════════════════════════════════════════════════════════════════
# Setup Daily Bot Export - Windows Task Scheduler
# ════════════════════════════════════════════════════════════════════════════

param(
    [switch]$Install,
    [switch]$Remove,
    [switch]$Test,
    [switch]$Status
)

$ProjectRoot = "C:\Users\DEVENDER\OneDrive\Desktop\voicbot\project-root"
$TaskName = "NSEIQ-Daily-Bot-Export"
$TaskDescription = "Daily export of NSEIQ trading bot data to CSV"
$ScriptPath = "$ProjectRoot\run_daily_export.bat"
$LogPath = "$ProjectRoot\logs\scheduled_task.log"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $Message"
    Add-Content -Path $LogPath -Value "[$timestamp] $Message" -ErrorAction SilentlyContinue
}

function Install-DailyTask {
    Write-Host "`n╔═════════════════════════════════════════════════════════════════════════════╗"
    Write-Host "║           📅 Setting up Daily Bot Export Scheduled Task                      ║"
    Write-Host "╚═════════════════════════════════════════════════════════════════════════════╝"
    
    # Create log directory
    $LogDir = Split-Path -Parent $LogPath
    if (-not (Test-Path $LogDir)) {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
        Write-Log "✅ Created log directory: $LogDir"
    }
    
    # Check if batch file exists
    if (-not (Test-Path $ScriptPath)) {
        Write-Host "`n❌ ERROR: Batch file not found at $ScriptPath"
        return $false
    }
    
    Write-Host "`n📋 Configuration:"
    Write-Host "   Task Name: $TaskName"
    Write-Host "   Script: $ScriptPath"
    Write-Host "   Schedule: Daily at 9:00 PM"
    Write-Host "   Frequency: Every day"
    
    # Check if task already exists
    $ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($ExistingTask) {
        Write-Host "`n⚠️  Task already exists. Removing old version..."
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Log "Removed existing task: $TaskName"
    }
    
    # Create task action
    $Action = New-ScheduledTaskAction `
        -Execute $ScriptPath `
        -WorkingDirectory $ProjectRoot
    
    # Create trigger (Daily at 9:00 PM)
    $Trigger = New-ScheduledTaskTrigger `
        -Daily `
        -At "21:00"  # 9:00 PM
    
    # Create task settings
    $Settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RunOnlyIfNetworkAvailable `
        -WakeToRun:$false
    
    # Register the task
    try {
        $Task = Register-ScheduledTask `
            -TaskName $TaskName `
            -Action $Action `
            -Trigger $Trigger `
            -Settings $Settings `
            -Description $TaskDescription `
            -ErrorAction Stop
        
        Write-Host "`n✅ Task created successfully!"
        Write-Host "   Next run: $(($Trigger.StartBoundary -as [datetime]).AddDays(1))"
        Write-Log "✅ Scheduled task created: $TaskName"
        
        return $true
    }
    catch {
        Write-Host "`n❌ Failed to create task: $_"
        Write-Log "❌ Failed to create task: $_"
        return $false
    }
}

function Remove-DailyTask {
    Write-Host "`n╔═════════════════════════════════════════════════════════════════════════════╗"
    Write-Host "║              ❌ Removing Daily Bot Export Scheduled Task                     ║"
    Write-Host "╚═════════════════════════════════════════════════════════════════════════════╝"
    
    $ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    
    if (-not $ExistingTask) {
        Write-Host "`n⚠️  Task not found: $TaskName"
        return $false
    }
    
    try {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
        Write-Host "`n✅ Task removed successfully!"
        Write-Log "✅ Scheduled task removed: $TaskName"
        return $true
    }
    catch {
        Write-Host "`n❌ Failed to remove task: $_"
        Write-Log "❌ Failed to remove task: $_"
        return $false
    }
}

function Test-DailyTask {
    Write-Host "`n╔═════════════════════════════════════════════════════════════════════════════╗"
    Write-Host "║          🧪 Testing Daily Bot Export (Running Immediately)                  ║"
    Write-Host "╚═════════════════════════════════════════════════════════════════════════════╝"
    
    Write-Host "`nRunning: $ScriptPath`n"
    
    & $ScriptPath
    
    Write-Host "`n✅ Test execution completed"
    Write-Log "✅ Test execution completed"
}

function Show-TaskStatus {
    Write-Host "`n╔═════════════════════════════════════════════════════════════════════════════╗"
    Write-Host "║              📊 Daily Bot Export Task Status                                 ║"
    Write-Host "╚═════════════════════════════════════════════════════════════════════════════╝"
    
    $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    
    if (-not $Task) {
        Write-Host "`n❌ Task not found: $TaskName`n"
        Write-Host "To install, run:"
        Write-Host "   powershell -ExecutionPolicy Bypass -File setup_daily_export.ps1 -Install"
        return
    }
    
    Write-Host "`n✅ Task Found: $TaskName"
    Write-Host "   Status: $($Task.State)"
    Write-Host "   Path: $($Task.TaskPath)"
    
    $LastTask = Get-ScheduledTaskInfo -TaskName $TaskName
    Write-Host "   Last Run: $($LastTask.LastRunTime)"
    Write-Host "   Last Result: $($LastTask.LastTaskResult)"
    Write-Host "   Next Run: $($LastTask.NextRunTime)"
    
    Write-Host "`n📋 Available Commands:"
    Write-Host "   Install: powershell -File setup_daily_export.ps1 -Install"
    Write-Host "   Remove:  powershell -File setup_daily_export.ps1 -Remove"
    Write-Host "   Test:    powershell -File setup_daily_export.ps1 -Test"
    Write-Host "   Status:  powershell -File setup_daily_export.ps1 -Status"
}

# Main logic
function Main {
    if ($Status) {
        Show-TaskStatus
    }
    elseif ($Test) {
        Test-DailyTask
    }
    elseif ($Install) {
        Install-DailyTask
    }
    elseif ($Remove) {
        Remove-DailyTask
    }
    else {
        Write-Host "`n╔═════════════════════════════════════════════════════════════════════════════╗"
        Write-Host "║         🤖 NSEIQ Daily Bot Export - Task Scheduler Setup                    ║"
        Write-Host "╚═════════════════════════════════════════════════════════════════════════════╝"
        Write-Host "`nUsage: powershell -ExecutionPolicy Bypass -File setup_daily_export.ps1 [Option]"
        Write-Host "`nOptions:"
        Write-Host "  -Install    : Setup daily export scheduled task (runs at 9:00 PM)"
        Write-Host "  -Remove     : Remove scheduled task"
        Write-Host "  -Test       : Run export immediately for testing"
        Write-Host "  -Status     : Show task status and logs"
        Write-Host "`nExamples:"
        Write-Host "  powershell -ExecutionPolicy Bypass -File setup_daily_export.ps1 -Install"
        Write-Host "  powershell -ExecutionPolicy Bypass -File setup_daily_export.ps1 -Test"
        Write-Host "  powershell -ExecutionPolicy Bypass -File setup_daily_export.ps1 -Status"
        Write-Host "`n"
    }
}

# Run main
Main
