"""Сборка: page5.html → artifact.html (без обёртки) и site/index.html (полный HTML)."""
import os
page = open("page5.html", encoding="utf-8").read()
open("artifact.html", "w", encoding="utf-8").write(page)
head, body = page.split('<div class="shell">', 1)
os.makedirs("site", exist_ok=True)
open("site/index.html", "w", encoding="utf-8").write(
    '<!doctype html>\n<html lang="ru">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    + head + '</head>\n<body>\n<div class="shell">' + body + '\n</body>\n</html>\n')
print("built", len(page) // 1024, "KB")
