# 一键备份到阿里云 OSS

在项目目录运行（需要 Python 3.9+ 和 pip）：

```bash
./backup_to_oss.sh
```

默认按香港时区的当天日期逐文件上传，保留项目中的相对路径：

```text
oss://jflyer/LooM/20271104/
  README.md
  .git/...
  datasets/...
  oss-backup-current/...
  scripts/...
  __loom_backup__/manifest.json
```

不生成压缩包。每个日期目录是一份完整的逻辑备份，未变化的同日文件可直接复用。

## 日期与重复运行

- 不同日期存放于不同目录，例如 `LooM/20271104/`、`LooM/20271105/`。
- 同一天重跑更新同名文件，核对大小与 CRC64 后跳过相同对象。
- 不删除 OSS 上的旧文件。因此同一天本地删除或改名后，旧对象仍可能存在；最新成功清单只列出本次备份中的路径。
- 失败后重跑会继续原日期的任务，即使已经跨过午夜；重新扫描当前源目录，大文件使用分片续传。
- 可用 `--date 20271104` 指定日期。已有待续传任务时，需先完成原任务，或指定另一个 `--work-dir`。
- 备份开始时将远端清单标记为 `in_progress`，全部成功后标记为 `complete`。请只使用 `complete` 状态的备份恢复。
- 同一天的目录在上传过程中会被更新；需要保留当天多份独立备份时，可使用不同的 `--prefix`。
- 上传期间应暂停修改数据的任务；扫描后或上传期间检测到文件变化会报错，重新运行可重试。请避免多台机器同时写入同一日期目录。

## 凭据

首次运行提示输入 AccessKey ID、Secret 和可选 STS Token，保存至
`~/.ossutilconfig`，权限为 `600`；密钥不写入脚本，也不出现在命令行参数中。
也可提前配置：

```bash
./backup_to_oss.sh --configure
```

已有 ossutil `[Credentials]` 配置可直接使用。环境变量 `OSS_ACCESS_KEY_ID`、
`OSS_ACCESS_KEY_SECRET` 优先于配置文件；临时凭据另设 `OSS_SESSION_TOKEN`。
需要目标前缀的上传、分片上传、查询分片和读取对象元数据权限。
首次上传会把 `oss2==2.19.1` 安装到缓存目录，不修改项目依赖。

## 内容与校验

- 包含 `.git`、Git 忽略的数据集、隐藏文件和 `oss-backup-current/`，不按 `.gitignore` 排除数据。
- 排除 `.cache`、虚拟环境、Python/测试缓存、`.aws`、`.ossutilconfig`、指定凭据文件和本脚本的缓存目录。
- `.env` 等项目配置也会上传，应按私有数据管理。
- 符号链接保存为 `<链接名>.rclonelink`，内容为链接目标；不跟随链接上传外部文件。这与此前备份中的 rclone 链接格式一致。
- 空目录保存为以 `/` 结尾的空对象。文件权限、修改时间和链接信息写入清单；OSS 对象本身不保留 POSIX 文件系统属性。
- 清单中的普通文件和链接对象均核对大小及 CRC64，并记录 SHA256；所有文件成功才发布 `complete` 清单。
- `__loom_backup__/` 为脚本保留的元信息目录。
- 本地缓存为 `.cache/oss-backup/`，含预览、续传信息和最近成功报告，不复制整份项目或创建压缩包。

## 常用选项

```bash
# 预览清单和目标路径，不读取凭据、不安装 SDK、不访问 OSS
./backup_to_oss.sh --dry-run

# 指定日期
./backup_to_oss.sh --date 20271104

# 仅备份之前下载的完整项目
./backup_to_oss.sh --source ./oss-backup-current

# 自定义来源和目标；--threads 为同时处理的文件数
./backup_to_oss.sh --source /path/to/LooM \
  --work-dir /path/to/backup-cache \
  --bucket jflyer --prefix LooM --threads 8

./backup_to_oss.sh --help
```

旧版压缩包任务若仍有 `pending.json`，脚本会停止并提示，不会删除旧备份包。
使用其他 `--work-dir` 可开始新的目录备份。

恢复时先检查日期目录下 `__loom_backup__/manifest.json` 为 `complete`，按其中的对象清单下载，
核对 SHA256，再按清单恢复文件权限、修改时间和符号链接。
普通文件也可直接从 OSS 控制台按目录下载；`.rclonelink` 需转换为符号链接，不能当作普通源码使用。

上传与校验使用阿里云官方 SDK 的
[断点续传](https://www.alibabacloud.com/help/en/oss/developer-reference/resumable-upload-1)
和 [CRC64 校验](https://www.alibabacloud.com/help/en/oss/developer-reference/data-verification-4)。
