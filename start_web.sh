#!/bin/bash

echo "=========================================="
echo "🚀 Professor Search Engine - Web Interface"
echo "=========================================="
echo ""

# Check if Flask is installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "⚠️  Flask 未安装，正在安装依赖..."
    pip3 install -r scripts/requirements.txt
    echo ""
fi

echo "启动 Web 服务器..."
echo "访问地址: http://localhost:5001"
echo ""
echo "按 Ctrl+C 停止服务器"
echo "=========================================="
echo ""

python3 app.py

