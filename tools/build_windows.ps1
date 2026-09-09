param(
    [string]$VcpkgRoot = $env:VCPKG_ROOT,
    [string]$Python = 'python',
    [string]$Generator = 'Visual Studio 18 2026',
    [switch]$InstallDependencies,
    [switch]$OneFile
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}
if (-not $VcpkgRoot -or -not (Test-Path -LiteralPath (Join-Path $VcpkgRoot 'vcpkg.exe'))) {
    throw 'Set VCPKG_ROOT or pass -VcpkgRoot to an existing vcpkg installation.'
}
if ($InstallDependencies) {
    Invoke-Checked (Join-Path $VcpkgRoot 'vcpkg.exe') @('install','pcl[core]:x64-windows','opencv4[core,calib3d,jpeg,png]:x64-windows','yaml-cpp:x64-windows','--recurse')
}
$sources = @(
    @{Name='Sophus'; Url='https://github.com/strasdat/Sophus.git'; Commit='a621ff2e56c56c839a6c40418d42c3c254424b5c'},
    @{Name='rpg_vikit'; Url='https://github.com/xuankuzcr/rpg_vikit.git'; Commit='6c886c8e5d83997806e00294826d528cea3581dd'}
)
foreach ($dependency in $sources) {
    $sourcePath = Join-Path $projectRoot ('build/deps-src/' + $dependency.Name)
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        Invoke-Checked git @('clone',$dependency.Url,$sourcePath)
        Invoke-Checked git @('-C',$sourcePath,'checkout',$dependency.Commit)
    }
    $actual = & git -C $sourcePath rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $actual.Trim() -ne $dependency.Commit) {
        throw "Unexpected revision in $sourcePath. Expected $($dependency.Commit); preserve any local changes before updating it."
    }
}
Invoke-Checked cmake @('-S','native','-B','build/native','-G',$Generator,'-A','x64',('-DCMAKE_TOOLCHAIN_FILE=' + (Join-Path $VcpkgRoot 'scripts/buildsystems/vcpkg.cmake')))
Invoke-Checked cmake @('--build','build/native','--config','Release','--parallel','4')
if (-not (Test-Path -LiteralPath 'build/portable-env/Scripts/python.exe')) {
    Invoke-Checked $Python @('-m','venv','build/portable-env')
}
$buildPython = Join-Path $projectRoot 'build/portable-env/Scripts/python.exe'
Invoke-Checked $buildPython @('-m','pip','install','-r','requirements-build.txt')
$packageArguments = @('tools/package_app.py','--vcpkg-root',$VcpkgRoot)
if ($OneFile) { $packageArguments += '--onefile' }
Invoke-Checked $buildPython $packageArguments
$binaryRelativePath = if ($OneFile) { 'dist/single-file/RavenCalibrator.exe' } else { 'dist/RavenCalibrator/RavenCalibrator.exe' }
Invoke-Checked (Join-Path $projectRoot $binaryRelativePath) @('--headless','doctor')
