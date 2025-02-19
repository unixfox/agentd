from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Header, Footer, Input, Static
from textual.scroll_view import ScrollView
from textual.screen import Screen
from textual import events
from textual.message import Message

# Import your AssistantManager from your library.
from astra_assistants.astra_assistants_manager import AssistantManager

from agentd.ui.assistant_util import create_manager, list_threads, list_messages


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


class ThreadListScreen(Screen):
    """Screen to display a searchable list of threads in a DataTable."""
    BINDINGS = [
        ("j", "table_down", "Move down"),
        ("k", "table_up", "Move up"),
        ("/", "focus_search", "Focus search"),
        ("l", "open_thread", "Open thread")
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Input(placeholder="Search threads...", id="search-input")
        yield MyDataTable(id="thread-table")
        yield Footer()

    def on_mount(self) -> None:
       # Initialize the assistant manager with your actual instructions.
        self.assistant_manager = create_manager(
            instructions="Your assistant instructions go here",
            model="gpt-4o"
        )
        try:
            threads = list_threads(self.assistant_manager)
            if not threads:
                raise Exception("No threads found.")
        except Exception as e:
            # Fallback: use the single thread from the manager.
            threads = [self.assistant_manager.thread]
        self.all_threads = threads
        self.load_table(self.all_threads)

    def load_table(self, threads_data):
        table: DataTable = self.query_one("#thread-table", DataTable)
        table.clear(columns=True)
        # For example, display the thread id and creation time.
        table.add_columns("ID", "Created At", "Metadata", "Last Message")
        for thread in threads_data:
            table.add_row(
                thread.id,
                str(thread.created_at),
                str(thread.metadata) if thread.metadata is not None else "N/A",
                str(thread.messages[0]) if len(thread.messages) else "N/A"
            )
        # Only shift focus if the search input is not active.
        search_input = self.query_one("#search-input", Input)
        if not search_input.has_focus:
            table.focus()

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "search-input":
            query = event.value.lower()
            filtered_threads = [
                t for t in self.all_threads
                if query in t.id.lower() or query in str(t.created_at).lower() or (t.metadata and query in str(t.metadata).lower())
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

    async def action_open_thread(self) -> None:
        table: DataTable = self.query_one("#thread-table", DataTable)
        row_index = table.cursor_row
        if row_index is not None:
            row_data = table.get_row_at(row_index)
            thread_id = row_data[0]
            # Since thread is a Pydantic model, compare using attribute access.
            thread = next((t for t in self.all_threads if t.id == thread_id), None)
            if thread:
                # Pass the assistant manager to the detail screen.
                await self.app.push_screen(ThreadDetailScreen(thread, manager=self.assistant_manager))

    async def on_thread_open_request(self, message: ThreadOpenRequest) -> None:
        await self.action_open_thread()


class ThreadDetailScreen(Screen):
    """Screen to display messages for a selected thread with live scrolling chat."""
    BINDINGS = [
        ("q", "pop_screen", "Back"),
        ("j", "scroll_down", "Scroll Down"),
        ("k", "scroll_up", "Scroll Up")
    ]

    def __init__(self, thread_data, manager, **kwargs):
        super().__init__(**kwargs)
        self.thread_data = thread_data  # A Pydantic model instance.
        self.manager = manager          # AssistantManager instance.

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(f"Thread: {self.thread_data.id}", id="thread-header")
        # Create a ScrollView that contains a Static widget (the actual log)
        yield ScrollView(Static(self.format_messages(), id="messages-content"), id="thread-messages")
        yield Input(placeholder="Type your message here...", id="chat-input")
        yield Footer()

    def format_messages(self) -> str:
        # Assume that the thread model now has an attribute "messages" (a list)
        messages = getattr(self.thread_data, "messages", [])
        if messages:
            # If messages are objects, access their attributes (adjust as needed)
            return "\n".join(f"[{msg.sender}] {msg.message}" for msg in messages)
        return "No messages yet."

    async def on_mount(self) -> None:
        # Retrieve messages for this thread via the API.
        messages = list_messages(self.manager, self.thread_data.id)
        # Update the thread model with these messages.
        self.thread_data.messages = messages
        # Instead of calling update() on the ScrollView, update its child Static widget.
        messages_widget: Static = self.query_one("#messages-content", Static)
        messages_widget.update(self.format_messages())

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat-input":
            user_input = event.value
            # Append the user's message.
            if not hasattr(self.thread_data, "messages") or self.thread_data.messages is None:
                self.thread_data.messages = []
            self.thread_data.messages.append({"sender": "User", "message": user_input})
            # Simulate an assistant reply.
            assistant_reply = f"Assistant reply to: {user_input}"
            self.thread_data.messages.append({"sender": "Assistant", "message": assistant_reply})
            # Update the child Static widget.
            messages_widget: Static = self.query_one("#messages-content", Static)
            messages_widget.update(self.format_messages())
            # Optionally, scroll to the end.
            scroll_view = self.query_one("#thread-messages", ScrollView)
            scroll_view.scroll_end()
            event.input.value = ""

    def action_scroll_down(self) -> None:
        scroll_view: ScrollView = self.query_one("#thread-messages", ScrollView)
        scroll_view.scroll_to(y=scroll_view.virtual_size.height, animate=True)

    def action_scroll_up(self) -> None:
        scroll_view: ScrollView = self.query_one("#thread-messages", ScrollView)
        scroll_view.scroll_to(y=0, animate=True)

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
