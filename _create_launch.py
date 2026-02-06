import json, os

os.makedirs('.vscode', exist_ok=True)

config = {
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug app.py (uv)",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/app.py",
            "console": "integratedTerminal",
            "python": "${workspaceFolder}/.venv/Scripts/python.exe",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            },
            "envFile": "${workspaceFolder}/.env"
        }
    ]
}

with open('.vscode/launch.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=4)

print("launch.json created, size:", os.path.getsize('.vscode/launch.json'))
