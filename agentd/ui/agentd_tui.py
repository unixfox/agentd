from datetime import datetime
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Header, Footer, Input
from textual.scroll_view import ScrollView
from textual.screen import Screen
from textual import events
from textual.message import Message

# Import your AssistantManager from your library.
from astra_assistants.astra_assistants_manager import AssistantManager
from agentd.ui.assistant_util import create_manager, list_threads, list_messages

import logging

logging.basicConfig(
    filename="agentd_tui.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

class ThreadOpenRequest(Message):
    def __init__(self, row_index: int) -> None:
        self.row_index = row_index
        super().__init__()

class MyDataTable(DataTable):
    def action_select_cursor(self) -> None:
        pass

    def key_enter(self) -> None:
        self.post_message(ThreadOpenRequest(self.cursor_row))

class ThreadListScreen(Screen):
    BINDINGS = [
        ("j", "table_down", "Move down"),
        ("k", "table_up", "Move up"),
        ("/", "focus_search", "Focus search"),
        ("l", "open_thread", "Open thread"),
        ("w", "toggle_wrap", "Toggle text wrap")
    ]

    def __init__(self, name: str | None = None, id: str | None = None, classes: str | None = None):
        super().__init__(name, id, classes)
        self.all_threads = None
        self.wrap_text = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Input(placeholder="Search threads...", id="search-input")
        yield MyDataTable(id="thread-table")
        yield Footer()

    def on_mount(self) -> None:
        self.assistant_manager = create_manager(
            instructions="Your assistant instructions go here",
            model="gpt-4o"
        )
        try:
            threads = list_threads(self.assistant_manager)
            if not threads:
                raise Exception("No threads found.")
        except Exception as e:
            threads = [self.assistant_manager.thread]
        self.all_threads = threads
        self.load_table(self.all_threads)

    def load_table(self, threads_data):
        sorted_threads = sorted(threads_data, key=lambda thread: thread.created_at, reverse=True)
        table: DataTable = self.query_one("#thread-table", DataTable)
        table.clear(columns=True)
        columns = ["ID", "Created At", "Metadata", "Last Message", "Assistant ID", "Role", "Run ID", "Status", "Message ID"]
        table.add_columns(*columns)

        for thread in sorted_threads:
            row_data = [
                thread.id,
                str(datetime.fromtimestamp(thread.created_at/1000)),
                str(thread.metadata) if thread.metadata is not None else "N/A",
                str(thread.messages[0].content[0].text.value) if thread.messages and len(thread.messages) else "N/A",
                str(thread.messages[0].assistant_id) if len(thread.messages) else "N/A",
                str(thread.messages[0].role) if len(thread.messages) else "N/A",
                str(thread.messages[0].run_id) if len(thread.messages) else "N/A",
                str(thread.messages[0].status) if len(thread.messages) else "N/A",
                str(thread.messages[0].id) if len(thread.messages) else "N/A"
            ]
            table.add_row(*row_data)

        self.update_table_wrapping(table)
        search_input = self.query_one("#search-input", Input)
        if not search_input.has_focus:
            table.focus()

    def update_table_wrapping(self, table: DataTable):
        """Update table column widths based on wrap_text state, checking column existence."""
        if 3 in table.columns:  # "Last Message"
            table.columns[3]._width = None if self.wrap_text else 30
        if 2 in table.columns:  # "Metadata"
            table.columns[2]._width = None if self.wrap_text else 20
        table.refresh()

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "search-input":
            query = event.value.lower()
            if not query:
                # If search is cleared, show all threads
                self.load_table(self.all_threads)
            else:
                filtered_threads = [
                    t for t in self.all_threads
                    if query in t.id.lower() or
                       query in str(t.created_at).lower() or
                       (t.metadata and query in str(t.metadata).lower()) or
                       (t.messages and len(t.messages) and query in str(t.messages[0].content[0].text.value).lower())
                ]
                self.load_table(filtered_threads)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "search-input":
            table: DataTable = self.query_one("#thread-table", DataTable)
            table.focus()

    def action_table_down(self) -> None:
        table: DataTable = self.query_one("#thread-table", DataTable)
        table.action_cursor_down()

    def action_table_up(self) -> None:
        table: DataTable = self.query_one("#thread-table", DataTable)
        table.action_cursor_up()

    def action_focus_search(self) -> None:
        search_input = self.query_one("#search-input", Input)
        search_input.focus()

    def action_toggle_wrap(self) -> None:
        self.wrap_text = not self.wrap_text
        table: DataTable = self.query_one("#thread-table", DataTable)
        self.update_table_wrapping(table)

    async def action_open_thread(self) -> None:
        table: DataTable = self.query_one("#thread-table", DataTable)
        row_index = table.cursor_row
        if row_index is not None:
            row_data = table.get_row_at(row_index)
            thread_id = row_data[0]
            thread = next((t for t in self.all_threads if t.id == thread_id), None)
            if thread:
                await self.app.push_screen(ThreadDetailScreen(thread, manager=self.assistant_manager))

    async def on_thread_open_request(self, message: ThreadOpenRequest) -> None:
        await self.action_open_thread()

class ThreadDetailScreen(Screen):
    BINDINGS = [
        ("q", "pop_screen", "Back"),
        ("escape", "pop_screen", "Back"),
        ("j", "table_down", "Move down"),
        ("k", "table_up", "Move up"),
        ("/", "focus_search", "Focus search"),
        ("w", "toggle_wrap", "Toggle text wrap")
    ]

    def __init__(self, thread_data, manager, **kwargs):
        super().__init__(**kwargs)
        self.thread_data = thread_data
        self.manager = manager
        self.all_messages = []
        self.wrap_text = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Input(placeholder="Search messages or type your message here...", id="search-input")
        yield ScrollView(MyDataTable(id="messages-table"), id="thread-messages")
        yield Footer()

    def load_messages(self, messages=None):
        if messages is None:
            messages = self.all_messages
        sorted_messages = sorted(messages, key=lambda msg: getattr(msg, 'created_at', 0))
        table: DataTable = self.query_one("#messages-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Role", "Message")
        for msg in sorted_messages:
            table.add_row(
                msg.role,
                msg.content[0].text.value if msg.content else "N/A"
            )
        scroll_view: ScrollView = self.query_one("#thread-messages", ScrollView)
        scroll_view.scroll_to(y=scroll_view.virtual_size.height, animate=False)
        self.update_table_wrapping(table)
        if not self.query_one("#search-input", Input).has_focus:
            table.focus()

    def update_table_wrapping(self, table: DataTable):
        """Update table column widths based on wrap_text state, checking column existence."""
        if 1 in table.columns:  # "Message"
            table.columns[1]._width = None if self.wrap_text else 50
        table.refresh()

    async def on_mount(self) -> None:
        messages = list_messages(self.manager, self.thread_data.id)
        self.thread_data.messages = messages
        self.all_messages = messages
        self.load_messages()

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "search-input":
            query = event.value.lower()
            if not query:
                # If search is cleared, show all messages
                self.load_messages(self.all_messages)
            else:
                filtered_messages = [
                    msg for msg in self.all_messages
                    if query in msg.role.lower() or
                       (msg.content and query in msg.content[0].text.value.lower())
                ]
                self.load_messages(filtered_messages)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "search-input":
            table: DataTable = self.query_one("#messages-table", DataTable)
            table.focus()

    def action_table_down(self) -> None:
        table: DataTable = self.query_one("#messages-table", DataTable)
        table.action_cursor_down()
        scroll_view: ScrollView = self.query_one("#thread-messages", ScrollView)
        cursor_y = table.cursor_coordinate.row
        scroll_view.scroll_to(y=cursor_y, animate=True)

    def action_table_up(self) -> None:
        table: DataTable = self.query_one("#messages-table", DataTable)
        table.action_cursor_up()
        scroll_view: ScrollView = self.query_one("#thread-messages", ScrollView)
        cursor_y = table.cursor_coordinate.row
        scroll_view.scroll_to(y=cursor_y, animate=True)

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def action_focus_search(self) -> None:
        search_input = self.query_one("#search-input", Input)
        search_input.focus()

    def action_toggle_wrap(self) -> None:
        self.wrap_text = not self.wrap_text
        table: DataTable = self.query_one("#messages-table", DataTable)
        self.update_table_wrapping(table)

class AssistantTUI(App):
    def on_mount(self) -> None:
        self.push_screen(ThreadListScreen())

if __name__ == "__main__":
    AssistantTUI().run()