#!/usr/bin/env pwsh

# $args[0] = Release/Debug
# $args[1] = other cmake defines

[CmdletBinding()]
Param
(
  [parameter(mandatory=$true, position=0)][string]$build_type,
  [parameter(mandatory=$false, position=1, ValueFromRemainingArguments=$true)]$other_cmake_flags
)

# Disable parallel building to avoid possible "CMake error : Cannot restore timestamp"
#$number_of_build_workers=(Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors

if (Get-Command "cl.exe" -ErrorAction SilentlyContinue) {
  $vstype = "Professional"
  if (Test-Path "C:\Program Files (x86)\Microsoft Visual Studio\2019\${vstype}\Common7\Tools") {
  }
  else {
    $vstype = "Enterprise"
    if (Test-Path "C:\Program Files (x86)\Microsoft Visual Studio\2019\${vstype}\Common7\Tools") {
    }
    else {
      $vstype = "Community"
    }
  }
  Write-Host "Found VS 2019 ${vstype}" -ForeGroundColor Yellow
  Push-Location "C:\Program Files (x86)\Microsoft Visual Studio\2019\${vstype}\Common7\Tools"
  cmd /c "VsDevCmd.bat -arch=x64 & set" |
    ForEach-Object {
    if ($_ -match "=") {
      $v = $_.split("="); set-item -force -path "ENV:\$($v[0])"  -value "$($v[1])"
    }
  }
  Pop-Location
  Write-Host "Visual Studio 2019 ${vstype} Command Prompt variables set.`n" -ForeGroundColor Yellow
}
else {
  Write-Host "No Compiler found" -ForeGroundColor Red
}

Push-Location $PSScriptRoot

If ( $build_type -eq $null )
{
  # DEBUG
  #Remove-Item .\build_win_debug -Force -Recurse -ErrorAction SilentlyContinue
  New-Item -Path .\build_win_debug -ItemType directory -Force
  Set-Location build_win_debug
  cmake -G "Visual Studio 16 2019" -T "host=x64" -A "x64" "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" "-DVCPKG_TARGET_TRIPLET=$env:VCPKG_DEFAULT_TRIPLET" "-DCMAKE_BUILD_TYPE:STRING=Debug" ${other_cmake_flags} ..
  cmake --build . --config Debug --target install
  Set-Location ..

  # RELEASE
  #Remove-Item .\build_win_release -Force -Recurse -ErrorAction SilentlyContinue
  New-Item -Path .\build_win_release -ItemType directory -Force
  Set-Location build_win_release
  cmake -G "Visual Studio 16 2019" -T "host=x64" -A "x64" "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" "-DVCPKG_TARGET_TRIPLET=$env:VCPKG_DEFAULT_TRIPLET" "-DCMAKE_BUILD_TYPE:STRING=Release" ${other_cmake_flags} ..
  cmake --build . --config Release --target install
  Set-Location ..
}
ElseIf ( $build_type -eq "Debug" -or $build_type -eq "debug" )
{
  # DEBUG
  #Remove-Item .\build_win_debug -Force -Recurse -ErrorAction SilentlyContinue
  New-Item -Path .\build_win_debug -ItemType directory -Force
  Set-Location build_win_debug
  cmake -G "Visual Studio 16 2019" -T "host=x64" -A "x64" "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" "-DVCPKG_TARGET_TRIPLET=$env:VCPKG_DEFAULT_TRIPLET" "-DCMAKE_BUILD_TYPE:STRING=Debug" ${other_cmake_flags} ..
  cmake --build . --config Debug --target install
  Set-Location ..
}
ElseIf ( $build_type -eq "Release" -or $build_type -eq "release" )
{
  # RELEASE
  #Remove-Item .\build_win_release -Force -Recurse -ErrorAction SilentlyContinue
  New-Item -Path .\build_win_release -ItemType directory -Force
  Set-Location build_win_release
  cmake -G "Visual Studio 16 2019" -T "host=x64" -A "x64" "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" "-DVCPKG_TARGET_TRIPLET=$env:VCPKG_DEFAULT_TRIPLET" "-DCMAKE_BUILD_TYPE:STRING=Release" ${other_cmake_flags} ..
  cmake --build . --config Release --target install
  Set-Location ..
}
Else
{
  Write-Host "Unknown build type - Allowed only [Debug, Release]" -ForeGroundColor Red
  exit 1
}


Pop-Location
