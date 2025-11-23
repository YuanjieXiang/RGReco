# 获取当前 Git 项目名（文件夹名）
$projectName = Split-Path -Path (Get-Location) -Leaf

# 获取当前 commit 的 short hash
$commitHash = git rev-parse --short HEAD

# 定义输出 zip 文件名
$outputZip = "$projectName-$commitHash.zip"

# 使用 git archive 导出当前 HEAD 内容为 zip
git archive --format=zip --output=$outputZip HEAD

Write-Host "✅ 已导出到: $outputZip"