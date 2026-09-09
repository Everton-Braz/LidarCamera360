# Compile all GLSL compute shaders to SPIR-V bytecode using Vulkan SDK glslc
param (
    [string]$ShaderDir = "native/vulkan_colorizer/shaders"
)

$glslc = (Get-Command glslc -ErrorAction SilentlyContinue).Source
if (-not $glslc -and $env:VULKAN_SDK) {
    $glslc = Join-Path $env:VULKAN_SDK "Bin\glslc.exe"
}

if (-not $glslc -or -not (Test-Path $glslc)) {
    Write-Error "glslc compiler not found. Ensure Vulkan SDK is installed and VULKAN_SDK is set."
    exit 1
}

Write-Host "[*] Using glslc: $glslc"
$shaders = Get-ChildItem -Path $ShaderDir -Filter "*.comp"

foreach ($s in $shaders) {
    $spv = [System.IO.Path]::ChangeExtension($s.FullName, ".spv")
    Write-Host "  -> Compiling $($s.Name) to $([System.IO.Path]::GetFileName($spv))..."
    & $glslc -fshader-stage=compute "$($s.FullName)" -o "$spv"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to compile $($s.Name)"
        exit $LASTEXITCODE
    }
}

Write-Host "[+] All compute shaders compiled to SPIR-V successfully!"
