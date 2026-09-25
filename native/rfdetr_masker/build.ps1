param(
    [Parameter(Mandatory = $true)]
    [string]$TensorRTRoot,
    [string]$TensorRTRuntimeDir = "",
    [string]$Generator = "",
    [string]$BuildDirectory = "build/rfdetr-masker",
    [switch]$IncludeBuilderResource
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "../..")).Path
$sourceDir = Join-Path $repoRoot "native/rfdetr_masker"
$buildDir = Join-Path $repoRoot $BuildDirectory
$trtRootFull = (Resolve-Path -LiteralPath $TensorRTRoot).Path
$runtimeDir = $TensorRTRuntimeDir
if (-not $runtimeDir) {
    foreach ($relative in @("lib", "bin", "runtime/tensorrt_libs")) {
        $candidate = Join-Path $trtRootFull $relative
        if (Test-Path -LiteralPath (Join-Path $candidate "nvinfer_10.dll")) {
            $runtimeDir = $candidate
            break
        }
    }
}
if (-not $runtimeDir -or -not (Test-Path -LiteralPath (Join-Path $runtimeDir "nvinfer_10.dll"))) {
    throw "Supply -TensorRTRuntimeDir with the TensorRT 10 DLL directory."
}

$cmakeArgs = @(
    "-S", $sourceDir,
    "-B", $buildDir,
    "-A", "x64",
    "-DRFDTR_TENSORRT_ROOT=$trtRootFull",
    "-DRFDTR_TENSORRT_RUNTIME_DIR=$runtimeDir"
)
if ($Generator) { $cmakeArgs += @("-G", $Generator) }
if ($IncludeBuilderResource) {
    $cmakeArgs += "-DRFDTR_COPY_BUILDER_RESOURCE=ON"
}

& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed with exit code $LASTEXITCODE." }
& cmake --build $buildDir --config Release --target rfdetr-masker --parallel
if ($LASTEXITCODE -ne 0) { throw "RF-DETR masker build failed with exit code $LASTEXITCODE." }

Write-Host "Built $(Join-Path $buildDir 'Release/rfdetr-masker.exe')"
