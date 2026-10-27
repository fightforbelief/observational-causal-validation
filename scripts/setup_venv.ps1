$ErrorActionPreference = "Stop"
$root = Join-Path $env:USERPROFILE "Desktop\projects\p4-observational-causal-validation"
$py = "C:\Users\hanqi\AppData\Local\Programs\Python\Python312\python.exe"
Set-Location $root
if (-not (Test-Path "$root\.venv\Scripts\python.exe")) {
  & $py -m venv "$root\.venv"
}
# Official PyPI downloads stalled through the machine proxy on 2026-10-26.
# The Tsinghua mirror served the same pinned wheels.
& "$root\.venv\Scripts\python.exe" -m pip install --timeout 60 --retries 8 -i https://pypi.tuna.tsinghua.edu.cn/simple numpy==2.2.6 pandas==2.3.2 pyarrow==22.0.0 scipy==1.15.3 scikit-learn==1.6.1 statsmodels==0.14.5 matplotlib==3.10.5 PyYAML==6.0.2 pytest==8.4.2 rdata==0.11.2
