import gradio as gr

from session.manager import list_sessions

DEFAULT_REMOTE_STYLE_LABEL = "\u9ed8\u8ba4"
DEFAULT_REMOTE_STYLE_VALUE = "__remote_default__"



FEATURE_GUIDE = """### 当前系统支持的功能

1. 文档上传入库
   示例操作：在“知识库文档管理”里上传 PDF，然后点击“加载文档”

2. 查看知识库已有文档
   示例操作：展开“知识库文档管理”，查看“当前知识库文档”

3. 删除知识库文档
   示例操作：在“知识库文档管理”里选择文档后点击“删除文档”

4. 为当前会话选择要加载的文档
   示例操作：在“当前会话加载文档”里勾选 1 篇或多篇文档

5. 基于已选文档进行问答
   示例提问：
   - “总结这篇论文的核心贡献”
   - “这篇文章的方法流程是什么？”

6. 多文档对比分析
   示例提问：
   - “对比这两篇论文的研究动机和方法差异”
   - “这几篇文档里谁更适合做参考方案？”

7. 会话保存与恢复
   示例操作：新建会话、切换历史会话，系统会恢复该会话的聊天记录和已选文档

8. 删除会话
   示例操作：选中一个历史会话后点击“删除当前会话”

9. 基于当前文档整理笔记/图文
   示例提问：
   - “把这篇论文整理成笔记”
   - “生成一篇适合发小红书的图文笔记”

10. 封面风格图库与风格迁移
   示例操作：
   - 在“封面风格图库”里上传或删除风格图
   - 为当前会话选择 1 张风格图
   - 发布确认前回复“需要风格迁移”
"""


def create_gradio_ui(app_factory):
    state = {"app": None, "user_id": "pdf_user1"}

    def _session_choices(user_id: str):
        sessions = list_sessions(user_id)
        return [(f"{s['title']}  ({s['created_at'][:16]})", s["session_id"]) for s in sessions]

    def _append_chat(history: list | None, user_text: str, assistant_text: str):
        items = list(history or [])
        items.append({"role": "user", "content": user_text})
        items.append({"role": "assistant", "content": assistant_text})
        return items

    def _document_names():
        if state["app"] is None:
            return []
        return [doc["filename"] for doc in state["app"].list_available_documents()]

    def _style_names():
        if state["app"] is None:
            return []
        return [(DEFAULT_REMOTE_STYLE_LABEL, DEFAULT_REMOTE_STYLE_VALUE)] + [item["filename"] for item in state["app"].list_style_images()]

    def _style_delete_names():
        if state["app"] is None:
            return []
        return [item["filename"] for item in state["app"].list_style_images()]

    def _document_summary() -> str:
        if state["app"] is None:
            return "请先新建或选择会话。"

        docs = state["app"].list_available_documents()
        if not docs:
            return "知识库中暂无文档。"

        lines = [f"当前知识库共 {len(docs)} 篇文档："]
        for index, doc in enumerate(docs, 1):
            lines.append(
                f"{index}. {doc['title']} | 文件名：{doc['filename']} | "
                f"入库时间：{doc['date_added']} | 分块数：{doc['chunk_count']}"
            )
        return "\n".join(lines)

    def _session_label() -> str:
        if state["app"] is None:
            return "请新建或选择一个会话"
        return (
            f"当前会话：{state['app'].session_id} | "
            f"已选文档：{len(state['app'].loaded_docs)} 篇"
        )

    def _style_summary() -> str:
        if state["app"] is None:
            return "请先新建或选择会话。"

        items = state["app"].list_style_images()
        if not items:
            return "当前风格图库为空。"

        lines = [f"当前风格图库共 {len(items)} 张图片："]
        for index, item in enumerate(items, 1):
            lines.append(f"{index}. {item['filename']} | 上传时间：{item['uploaded_at']}")
        return "\n".join(lines)

    def _style_status_text() -> str:
        if state["app"] is None:
            return "??????????"
        selected = state["app"].selected_style_image
        if selected == DEFAULT_REMOTE_STYLE_VALUE:
            return "???????????????????? MCP ?????"
        if selected:
            return f"???????????{selected}"
        return "????????????????????????? MCP ?????"

    def _document_updates():
        names = _document_names()
        selected = [] if state["app"] is None else list(state["app"].loaded_docs)
        return (
            gr.update(choices=names, value=selected),
            gr.update(choices=names, value=None),
            _document_summary(),
            _session_label(),
        )

    def _style_updates():
        names = _style_names()
        delete_names = _style_delete_names()
        selected = None if state["app"] is None or not state["app"].selected_style_image else state["app"].selected_style_image
        return (
            gr.update(choices=names, value=selected),
            gr.update(choices=delete_names, value=None),
            _style_summary(),
            _style_status_text(),
        )

    def _resolve_post_session_change(status_text: str):
        sessions = _session_choices(state["user_id"])
        if sessions:
            next_session_id = sessions[0][1]
            state["app"] = app_factory(user_id=state["user_id"], session_id=next_session_id)
            doc_selector, delete_dropdown, doc_summary, session_label = _document_updates()
            style_selector, style_delete_dropdown, style_summary, style_status = _style_updates()
            return (
                gr.update(choices=sessions, value=next_session_id),
                state["app"].get_chat_history(),
                session_label,
                doc_selector,
                delete_dropdown,
                doc_summary,
                style_selector,
                style_delete_dropdown,
                style_summary,
                style_status,
                status_text,
            )

        state["app"] = None
        doc_selector, delete_dropdown, doc_summary, session_label = _document_updates()
        style_selector, style_delete_dropdown, style_summary, style_status = _style_updates()
        return (
            gr.update(choices=[], value=None),
            [],
            session_label,
            doc_selector,
            delete_dropdown,
            doc_summary,
            style_selector,
            style_delete_dropdown,
            style_summary,
            style_status,
            status_text,
        )

    def new_session(user_id: str):
        if not user_id.strip():
            user_id = "pdf_user1"
        state["user_id"] = user_id
        state["app"] = app_factory(user_id=user_id)
        doc_selector, delete_dropdown, doc_summary, session_label = _document_updates()
        style_selector, style_delete_dropdown, style_summary, style_status = _style_updates()
        return (
            gr.update(choices=_session_choices(user_id), value=state["app"].session_id),
            [],
            session_label,
            doc_selector,
            delete_dropdown,
            doc_summary,
            style_selector,
            style_delete_dropdown,
            style_summary,
            style_status,
            "已新建会话，请勾选本次会话要加载的文档。",
        )

    def load_session(session_id: str):
        if not session_id:
            empty_updates = _document_updates()
            empty_style_updates = _style_updates()
            return [], "请先选择会话", *empty_updates, *empty_style_updates, "请先选择会话。"
        state["app"] = app_factory(user_id=state["user_id"], session_id=session_id)
        doc_selector, delete_dropdown, doc_summary, session_label = _document_updates()
        style_selector, style_delete_dropdown, style_summary, style_status = _style_updates()
        return (
            state["app"].get_chat_history(),
            session_label,
            doc_selector,
            delete_dropdown,
            doc_summary,
            style_selector,
            style_delete_dropdown,
            style_summary,
            style_status,
            "已恢复该会话的文档选择。",
        )

    def refresh_list(user_id: str):
        if user_id.strip():
            state["user_id"] = user_id
        doc_selector, delete_dropdown, doc_summary, session_label = _document_updates()
        style_selector, style_delete_dropdown, style_summary, style_status = _style_updates()
        return (
            gr.update(choices=_session_choices(state["user_id"])),
            doc_selector,
            delete_dropdown,
            doc_summary,
            session_label,
            style_selector,
            style_delete_dropdown,
            style_summary,
            style_status,
        )

    def load_pdf(pdf_file):
        if state["app"] is None:
            doc_selector, delete_dropdown, doc_summary, session_label = _document_updates()
            style_selector, style_delete_dropdown, style_summary, style_status = _style_updates()
            return "请先新建或选择会话", doc_selector, delete_dropdown, doc_summary, session_label, style_selector, style_delete_dropdown, style_summary, style_status
        if pdf_file is None:
            doc_selector, delete_dropdown, doc_summary, session_label = _document_updates()
            style_selector, style_delete_dropdown, style_summary, style_status = _style_updates()
            return "请先上传 PDF 文件", doc_selector, delete_dropdown, doc_summary, session_label, style_selector, style_delete_dropdown, style_summary, style_status

        result = state["app"].load_document(pdf_file)
        doc_selector, delete_dropdown, doc_summary, session_label = _document_updates()
        style_selector, style_delete_dropdown, style_summary, style_status = _style_updates()
        message = f"✅ {result['message']}" if result["success"] else f"❌ {result['message']}"
        return message, doc_selector, delete_dropdown, doc_summary, session_label, style_selector, style_delete_dropdown, style_summary, style_status

    def update_selected_documents(selected_docs: list[str]):
        if state["app"] is None:
            doc_selector, delete_dropdown, doc_summary, session_label = _document_updates()
            return "请先新建或选择会话。", doc_selector, session_label, delete_dropdown, doc_summary

        selected = state["app"].set_session_documents(selected_docs or [])
        doc_selector, delete_dropdown, doc_summary, session_label = _document_updates()
        return (
            f"当前会话已选择 {len(selected)} 篇文档。",
            doc_selector,
            session_label,
            delete_dropdown,
            doc_summary,
        )

    def delete_document(filename: str):
        if state["app"] is None:
            doc_selector, delete_dropdown, doc_summary, session_label = _document_updates()
            style_selector, style_delete_dropdown, style_summary, style_status = _style_updates()
            return "请先新建或选择会话。", doc_selector, delete_dropdown, doc_summary, session_label, style_selector, style_delete_dropdown, style_summary, style_status

        result = state["app"].delete_document(filename)
        doc_selector, delete_dropdown, doc_summary, session_label = _document_updates()
        style_selector, style_delete_dropdown, style_summary, style_status = _style_updates()
        message = f"✅ {result['message']}" if result["success"] else f"❌ {result['message']}"
        return message, doc_selector, delete_dropdown, doc_summary, session_label, style_selector, style_delete_dropdown, style_summary, style_status

    def update_selected_style_image(filename: str):
        if state["app"] is None:
            style_selector, style_delete_dropdown, style_summary, style_status = _style_updates()
            return "??????????", style_selector, style_delete_dropdown, style_summary

        selected = state["app"].set_session_style_image(filename or "")
        style_selector, style_delete_dropdown, style_summary, style_status = _style_updates()
        if selected == DEFAULT_REMOTE_STYLE_VALUE:
            message = "???????????????????? MCP ?????"
        elif selected:
            message = f"???????????{selected}"
        else:
            message = "????????????????????????? MCP ?????"
        return message, style_selector, style_delete_dropdown, style_summary

    def upload_style_image(style_file):
        if state["app"] is None:
            style_selector, style_delete_dropdown, style_summary, style_status = _style_updates()
            return "请先新建或选择会话", style_selector, style_delete_dropdown, style_summary
        if style_file is None:
            style_selector, style_delete_dropdown, style_summary, style_status = _style_updates()
            return "请先上传图片文件", style_selector, style_delete_dropdown, style_summary

        result = state["app"].upload_style_image(style_file)
        style_selector, style_delete_dropdown, style_summary, style_status = _style_updates()
        message = f"✅ {result['message']}" if result["success"] else f"❌ {result['message']}"
        return message, style_selector, style_delete_dropdown, style_summary

    def delete_style_image(filename: str):
        if state["app"] is None:
            style_selector, style_delete_dropdown, style_summary, style_status = _style_updates()
            return "请先新建或选择会话。", style_selector, style_delete_dropdown, style_summary

        result = state["app"].delete_style_image(filename)
        style_selector, style_delete_dropdown, style_summary, style_status = _style_updates()
        message = f"✅ {result['message']}" if result["success"] else f"❌ {result['message']}"
        return message, style_selector, style_delete_dropdown, style_summary

    def delete_active_session():
        if state["app"] is None:
            doc_selector, delete_dropdown, doc_summary, session_label = _document_updates()
            style_selector, style_delete_dropdown, style_summary, style_status = _style_updates()
            return (
                gr.update(choices=_session_choices(state["user_id"])),
                [],
                session_label,
                doc_selector,
                delete_dropdown,
                doc_summary,
                style_selector,
                style_delete_dropdown,
                style_summary,
                style_status,
                "请先新建或选择会话。",
            )

        result = state["app"].delete_current_session()
        status = f"✅ {result['message']}" if result["success"] else f"❌ {result['message']}"
        if not result["success"]:
            doc_selector, delete_dropdown, doc_summary, session_label = _document_updates()
            style_selector, style_delete_dropdown, style_summary, style_status = _style_updates()
            return (
                gr.update(choices=_session_choices(state["user_id"])),
                state["app"].get_chat_history(),
                session_label,
                doc_selector,
                delete_dropdown,
                doc_summary,
                style_selector,
                style_delete_dropdown,
                style_summary,
                style_status,
                status,
            )

        return _resolve_post_session_change(status)

    def chat(message: str, history: list):
        if state["app"] is None:
            return "", _append_chat(history, message, "请先新建或选择会话")
        if not message.strip():
            return "", history
        response = state["app"].ask(message)
        return "", _append_chat(history, message, response)

    with gr.Blocks(title="Multi-Agent 文档问答助手") as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### 文档问答助手")
                user_id_input = gr.Textbox(
                    label="用户ID",
                    value="pdf_user1",
                    container=False,
                    placeholder="输入用户ID",
                )
                new_btn = gr.Button("新建对话", variant="primary", size="sm")
                refresh_btn = gr.Button("刷新列表", size="sm")

                session_radio = gr.Radio(
                    label="历史会话",
                    choices=_session_choices("pdf_user1"),
                    interactive=True,
                    elem_classes="session-list",
                )
                delete_session_btn = gr.Button("删除当前会话", variant="stop", size="sm")

                with gr.Accordion("系统支持功能与示例", open=False):
                    gr.Markdown(FEATURE_GUIDE)

                with gr.Accordion("当前会话加载文档", open=True):
                    doc_selector = gr.CheckboxGroup(
                        label="勾选本会话要参与问答的文档",
                        choices=[],
                        value=[],
                    )
                    selection_status = gr.Textbox(
                        label="会话文档状态",
                        interactive=False,
                        lines=2,
                    )

                with gr.Accordion("封面风格图库", open=False):
                    style_selector = gr.Dropdown(
                        label="为当前会话选择风格图",
                        choices=[],
                        value=None,
                    )
                    style_status = gr.Textbox(
                        label="风格图状态",
                        interactive=False,
                        lines=2,
                    )
                    style_upload = gr.File(
                        label="上传风格图",
                        file_types=["image"],
                        type="filepath",
                    )
                    style_upload_btn = gr.Button("保存风格图", size="sm")
                    style_delete_dropdown = gr.Dropdown(
                        label="选择要删除的风格图",
                        choices=[],
                        value=None,
                    )
                    style_delete_btn = gr.Button("删除风格图", variant="stop", size="sm")
                    style_summary = gr.Textbox(
                        label="当前风格图库",
                        interactive=False,
                        lines=8,
                    )

                with gr.Accordion("知识库文档管理", open=False):
                    pdf_upload = gr.File(
                        label="上传 PDF",
                        file_types=[".pdf"],
                        type="filepath",
                    )
                    load_btn = gr.Button("加载文档", size="sm")
                    load_output = gr.Textbox(label="上传状态", interactive=False, lines=2)
                    delete_dropdown = gr.Dropdown(
                        label="选择要删除的文档",
                        choices=[],
                        value=None,
                    )
                    delete_btn = gr.Button("删除文档", variant="stop", size="sm")
                    doc_summary = gr.Textbox(
                        label="当前知识库文档",
                        interactive=False,
                        lines=12,
                    )

            with gr.Column(scale=4):
                session_label = gr.Textbox(
                    value="请新建或选择一个会话",
                    interactive=False,
                    container=False,
                    show_label=False,
                )
                chatbot = gr.Chatbot(label="", height=520, show_label=False)
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="输入消息，按 Enter 发送...",
                        show_label=False,
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

        new_btn.click(
            new_session,
            inputs=[user_id_input],
            outputs=[
                session_radio,
                chatbot,
                session_label,
                doc_selector,
                delete_dropdown,
                doc_summary,
                style_selector,
                style_delete_dropdown,
                style_summary,
                style_status,
                selection_status,
            ],
        )
        refresh_btn.click(
            refresh_list,
            inputs=[user_id_input],
            outputs=[session_radio, doc_selector, delete_dropdown, doc_summary, session_label, style_selector, style_delete_dropdown, style_summary, style_status],
        )
        session_radio.change(
            load_session,
            inputs=[session_radio],
            outputs=[chatbot, session_label, doc_selector, delete_dropdown, doc_summary, style_selector, style_delete_dropdown, style_summary, style_status, selection_status],
        )
        delete_session_btn.click(
            delete_active_session,
            outputs=[
                session_radio,
                chatbot,
                session_label,
                doc_selector,
                delete_dropdown,
                doc_summary,
                style_selector,
                style_delete_dropdown,
                style_summary,
                style_status,
                selection_status,
            ],
        )
        doc_selector.change(
            update_selected_documents,
            inputs=[doc_selector],
            outputs=[selection_status, doc_selector, session_label, delete_dropdown, doc_summary],
        )
        style_selector.change(
            update_selected_style_image,
            inputs=[style_selector],
            outputs=[style_status, style_selector, style_delete_dropdown, style_summary],
        )
        style_upload_btn.click(
            upload_style_image,
            inputs=[style_upload],
            outputs=[style_status, style_selector, style_delete_dropdown, style_summary],
        )
        style_delete_btn.click(
            delete_style_image,
            inputs=[style_delete_dropdown],
            outputs=[style_status, style_selector, style_delete_dropdown, style_summary],
        )
        load_btn.click(
            load_pdf,
            inputs=[pdf_upload],
            outputs=[load_output, doc_selector, delete_dropdown, doc_summary, session_label, style_selector, style_delete_dropdown, style_summary, style_status],
        )
        delete_btn.click(
            delete_document,
            inputs=[delete_dropdown],
            outputs=[load_output, doc_selector, delete_dropdown, doc_summary, session_label, style_selector, style_delete_dropdown, style_summary, style_status],
        )
        msg_input.submit(chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
        send_btn.click(chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])

    return demo
