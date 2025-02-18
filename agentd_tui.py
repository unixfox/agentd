from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Header, Footer, Input, Static
from textual.scroll_view import ScrollView
from textual.screen import Screen
from textual import events
from textual.message import Message

# Define a custom message to signal that a thread should be opened.
class ThreadOpenRequest(Message):
    def __init__(self, row_index: int) -> None:
        self.row_index = row_index
        super().__init__()

# Subclass DataTable to override its default behavior.
class MyDataTable(DataTable):
    def action_select_cursor(self) -> None:
        # Override default behavior so that Enter does not select the cell.
        pass

    def key_enter(self) -> None:
        # When Enter is pressed, post a message with the current cursor row.
        self.post_message(ThreadOpenRequest(self.cursor_row))

# Sample thread data – in a real application this might come from a database or API.
THREADS = [
    {
        "id": "Thread-001",
        "recent_message": "Hello, how can I help you?",
        "recent_assistant": "Assistant-1",
        "age": "2h ago",
        "message_count": 10,
        "messages": [
            {"sender": "User", "message": "Hi"},
            {"sender": "Assistant", "message": "Hello, how can I help you?"},
        ],
    },
    {
        "id": "Thread-002",
        "recent_message": "Goodbye for now!",
        "recent_assistant": "Assistant-2",
        "age": "1d ago",
        "message_count": 5,
        "messages": [
            {"sender": "User", "message": "What's the weather like?"},
            {"sender": "Assistant", "message": "It's sunny!"},
        ],
    },
    # Add more threads as needed...
]

class ThreadListScreen(Screen):
    """Screen to display a searchable and sortable list of threads in a DataTable."""
    BINDINGS = [
        ("j", "table_down", "Move down"),
        ("k", "table_up", "Move up"),
        ("/", "focus_search", "Focus search"),
        ("l", "open_thread", "Open thread")
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        # Input for filtering/searching threads.
        yield Input(placeholder="Search threads...", id="search-input")
        yield MyDataTable(id="thread-table")
        yield Footer()

    def on_mount(self) -> None:
        self.all_threads = THREADS.copy()  # Preserve master data.
        self.load_table(self.all_threads)

    def load_table(self, threads_data):
        table: DataTable = self.query_one("#thread-table", DataTable)
        table.clear(columns=True)
        table.add_columns("ID", "Recent Message", "Recent Assistant", "Age", "Message Count")
        for thread in threads_data:
            table.add_row(
                thread["id"],
                thread["recent_message"],
                thread["recent_assistant"],
                thread["age"],
                str(thread["message_count"])
            )
        # Only change focus if the search input is not active.
        search_input = self.query_one("#search-input", Input)
        if not search_input.has_focus:
            table.focus()

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "search-input":
            query = event.value.lower()
            filtered_threads = [
                t for t in self.all_threads
                if query in t["id"].lower()
                   or query in t["recent_message"].lower()
                   or query in t["recent_assistant"].lower()
            ]
            self.load_table(filtered_threads)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        # When the user hits Enter in the search input, shift focus to the table.
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

    async def action_open_thread(self) -> None:
        table: DataTable = self.query_one("#thread-table", DataTable)
        row_index = table.cursor_row
        if row_index is not None:
            row_data = table.get_row_at(row_index)
            thread_id = row_data[0]
            thread = next((t for t in self.all_threads if t["id"] == thread_id), None)
            if thread:
                await self.app.push_screen(ThreadDetailScreen(thread))

    async def on_thread_open_request(self, message: ThreadOpenRequest) -> None:
        await self.action_open_thread()

class ThreadDetailScreen(Screen):
    """Screen to display messages for a selected thread with live scrolling chat."""
    BINDINGS = [
        ("q", "pop_screen", "Back"),
        ("j", "scroll_down", "Scroll Down"),
        ("k", "scroll_up", "Scroll Up")
    ]

    def __init__(self, thread_data: dict, **kwargs):
        super().__init__(**kwargs)
        self.thread_data = thread_data

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(f"Thread: {self.thread_data['id']}", id="thread-header")
        yield ScrollView(Static(self.format_messages(), id="messages-content"), id="thread-messages")
        yield Input(placeholder="Type your message here...", id="chat-input")
        yield Footer()

    def format_messages(self) -> str:
        messages = self.thread_data.get("messages", [])
        return "\n".join(f"[{msg['sender']}] {msg['message']}" for msg in messages)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat-input":
            user_input = event.value
            self.thread_data["messages"].append({"sender": "User", "message": user_input})
            assistant_reply = f"Assistant reply to: {user_input}"
            self.thread_data["messages"].append({"sender": "Assistant", "message": assistant_reply})
            messages_widget: ScrollView = self.query_one("#thread-messages", ScrollView)
            messages_widget.update(Static(self.format_messages(), id="messages-content"))
            event.input.value = ""

    def action_scroll_down(self) -> None:
        scroll_view: ScrollView = self.query_one("#thread-messages", ScrollView)
        scroll_view.scroll_down()

    def action_scroll_up(self) -> None:
        scroll_view: ScrollView = self.query_one("#thread-messages", ScrollView)
        scroll_view.scroll_up()

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    async def on_key(self, event: events.Key) -> None:
        if event.key in ("down", "j"):
            self.action_scroll_down()
        elif event.key in ("up", "k"):
            self.action_scroll_up()

class AssistantTUI(App):
    """Main Textual Application for the assistant TUI."""
    def on_mount(self) -> None:
        self.push_screen(ThreadListScreen())

if __name__ == "__main__":
    AssistantTUI().run()
