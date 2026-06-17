Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

Write-Host "VUI LONG CHUYEN SANG CUA SO STREAMLIT NGAY BAY GIO! Anh se duoc chup sau 5 giay..."
Start-Sleep -Seconds 5

$screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp = New-Object System.Drawing.Bitmap($screen.Width, $screen.Height)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($screen.Location, [System.Drawing.Point]::Empty, $screen.Size)
$bmp.Save("d:\Risk\reports\dashboard_screenshot.png")
$g.Dispose()
$bmp.Dispose()
[System.Console]::Beep(1000, 500)
Write-Host "Screenshot saved successfully"
