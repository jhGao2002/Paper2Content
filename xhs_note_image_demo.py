from __future__ import annotations

import json
from pathlib import Path

from config import get_llm
from xhs.note_service import XHSNoteService


def mock_conversation() -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                "我把几篇风格迁移论文都丢进系统了，但问“不同方法到底差在哪”，"
                "回答总是漂在表面，像把摘要重新说一遍。"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "这通常不是模型不会答，而是检索上下文没有把真正可比较的段落召回出来。"
                "如果只用固定切块，模型常常拿到零碎句子，所以只能给泛化回答。"
            ),
        },
        {
            "role": "user",
            "content": "那应该怎么改？我希望它能回答“Style Injection 和 Style Aligned 的核心差异”。",
        },
        {
            "role": "assistant",
            "content": (
                "可以把检索改成父子块：先用小块提高召回命中率，再回溯到父块给模型完整上下文。"
                "这样模型看到的是方法机制、限制和实验结论的整段信息，不是碎片。"
            ),
        },
        {
            "role": "user",
            "content": "只有父子块就够了吗？我的追问经常也会丢上下文。",
        },
        {
            "role": "assistant",
            "content": (
                "还要把会话记忆和最近检索结果保留下来。这样用户第二轮追问“那谁更适合参考图风格迁移”时，"
                "系统能继承上一轮的比较对象，而不是重新猜你的问题。"
            ),
        },
        {
            "role": "user",
            "content": "如果把这次排查经验整理成一篇小红书笔记，你会强调什么？",
        },
        {
            "role": "assistant",
            "content": (
                "我会强调一个结论：论文问答答偏，很多时候不是模型笨，而是检索颗粒度和会话承接没设计好。"
                "正文可以按“问题现象 - 为什么答偏 - 两个关键改动 - 改完后能回答什么问题”来写。"
            ),
        },
    ]


def main() -> None:
    output_dir = Path("result") / "xhs_note_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    service = XHSNoteService(llm=get_llm())
    artifact = service.generate_note_artifact(
        history=mock_conversation(),
        generate_images=True,
        image_count=1,
        output_dir=str(output_dir),
    )
    prepared = service.publish_generated_note(artifact)

    note = artifact["note"]
    markdown = (
        f"# {note['title']}\n\n"
        f"> {note['summary']}\n\n"
        f"{note['body']}\n\n"
        f"{note['cta']}\n\n"
        f"标签：{' '.join('#' + tag.lstrip('#') for tag in note['hashtags'])}\n"
    )

    (output_dir / "note.md").write_text(markdown, encoding="utf-8")
    (output_dir / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"标题：{note['title']}")
    if prepared.get("data"):
        print(f"上传标题：{prepared['data'].get('title', '')}")
        print(f"论文简称：{prepared['data'].get('paper_title_short', '')}")
    print(f"问题：{note['core_problem']}")
    print(f"解决：{note['solved_problem']}")
    print(f"封面 prompt：{artifact['image_prompt']}")
    if artifact["image_paths"]:
        print(f"已生成图片：{artifact['image_paths'][0]}")
    else:
        print(f"本次未生成图片：{artifact['image_error'] or '请补充图片 API 配置后重试'}")
    print(f"结果目录：{output_dir.resolve()}")


if __name__ == "__main__":
    main()
