@echo off

set GO_PATH="D:\Programs\go-cqhttp"
set BOT_PATH="F:\Project\paddle\novel_bot"

set go_cmd="cd /d %GO_PATH% && go-cqhttp_windows_amd64.exe"
set bot_cmd="cd /d %BOT_PATH% && activate paddle && python bot.py"

start cmd /k %go_cmd%
start cmd /k %bot_cmd%

::pause
