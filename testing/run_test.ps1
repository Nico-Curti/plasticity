#!/usr/bin/env pwsh

$test_files = Get-ChildItem -Path ./bin/ -Include test_*

Write-Host "Run testing" -ForegroundColor Yellow

For ($i = 0; $i -lt $test_files.Length; $i++)
{
  Write-Host -NoNewline "* Running $test_files[$i] ...       "

  Invoke-Expression $test_files[$i]

  If ( $? )
  {
    Write-Host "[failed]" -ForegroundColor Red
    exit 1
  }

}

Write-Host "PASSED" -ForegroundColor Yellow
