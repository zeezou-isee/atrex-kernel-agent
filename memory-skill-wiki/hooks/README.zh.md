# 实时监视 hook

[English](README.md) · **中文**

`memory_extract_hook.py` 是一个**非阻塞**的 `Stop`(及 `SubagentStop`)hook。它在每个
session 的每个回合边界都会触发,但**只对通过 `/arm-run` 打开监视的 session 真正做事**——
其它情况一律立即 `exit 0`(纯空操作,对 agent 不可见)。它与任何其它 Stop hook 共存。

## 每次 Stop 时做什么
1. 读 payload(`transcript_path`、`cwd`、`session_id`、`stop_hook_active`)。
2. **先查 armed 标记**:若 `state/armed/<session_id>.json` 不存在 → `exit 0`
   (就这一次 stat,保证对无关 session 零负担)。
3. **去抖**(`checkpoint.py should-run`):距上次抽取 ≥ 标记里的 `debounce_min`(默认 20)分钟。
4. **有新内容**(`detect_change.py`):满足任一 —— 新的 AKA `memory/vN.json`、性能/PASS-FAIL
   结果行、kernel 的 `git commit`、或 ≥ `min_turns` 个新回合。
5. 都满足则:冻结末尾 uuid,把 `run_extraction.sh` 完全分离地(独立会话)拉起,然后 `exit 0`。
   worker 把新切片提炼进 `knowledge/<operator>/`、重建 wiki、推进 checkpoint、提交——全程
   与正在运行的会话隔离。

## 安装
用顶层安装脚本(一次装好 skill + 命令 + 本 hook):

```bash
bash ../install.sh --global          # 或:--claude-dir <proj>/.claude
```

`atrex_settings_snippet.json` 里是原始的 `settings.json` 片段,想手动合并也可以。装完**重启**
运行时,hook 才会加载。

## 调参
- `MEMORY_SKILL_DEBOUNCE_MIN` —— 两次抽取的最小间隔分钟数(默认 20)。
- `MEMORY_SKILL_LLM_CMD` —— 覆盖 LLM 步骤(收到 filtered-md 路径作最后一个参数,须写出
  `knowledge/*.json`)。用于不依赖 `claude` 的测试。

## 卸载
`bash ../install.sh --uninstall` —— 移除标签为 `kernel-experience-memory-hook-v1` 的 hook
条目、命令文件,以及 skill 软链。
