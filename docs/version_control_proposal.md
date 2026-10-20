# TriWeave本地版本管理建议

本轮检查TriWeave仍不是Git仓库；未初始化、提交或推送。现有源码以SHA256和只读科研产物快照管理。

可在后续明确执行本地`git init`，先审查`git status --short --ignored`，仅显式加入src、tests、docs、公共configs、pyproject.toml、requirements.txt和.gitignore。不要使用未经审查的`git add .`。配置CSV的固定目标顺序应随协议版本管理，263只原始股票名单及来源文件继续保存在被排除的data目录。

现有.gitignore排除全部行情层、manifest、实验结果、日志、虚拟环境和模型文件。本轮补充.env.*、私钥和本地私人配置排除规则。排除规则不证明已经移除历史泄露；若以后建立仓库，首次提交前仍需检查实际暂存内容。没有自动添加GitHub remote，也没有提交或推送。
