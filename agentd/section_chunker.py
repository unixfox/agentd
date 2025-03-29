import re

# Define our section markers.
SECTION_START_FMT = "###SECTION:{section_id}###"
SECTION_END = "###ENDSECTION###"
section_start_pattern = r"^###SECTION:[^#]+###$"
section_end_pattern = r"^###ENDSECTION###$"

def split_into_chunks(text, max_chunk_size=400):
    """
    Splits the input text into chunks of at most max_chunk_size characters,
    preserving whole lines.
    """
    if text is None:
        return ""
    lines = text.splitlines()
    chunks = []
    current_chunk = ""
    for line in lines:
        # +1 for newline.
        if len(current_chunk) + len(line) + 1 <= max_chunk_size:
            current_chunk += line + "\n"
        else:
            chunks.append(current_chunk.rstrip())
            current_chunk = line + "\n"
    if current_chunk:
        chunks.append(current_chunk.rstrip())
    return chunks

def unwrap_sections(text):
    """
    Removes section markers from the given text.
    Assumes that markers appear on their own lines.
    """
    if text is None:
        return ""
    lines = text.splitlines()
    filtered_lines = [
        line for line in lines
        if not (re.match(section_start_pattern, line.strip()) or re.match(section_end_pattern, line.strip()))
    ]
    return "\n".join(filtered_lines)

def wrap_sections(text, max_chunk_size=400):
    """
    Wraps the given file content into sections using our markers.
    Each section is assigned a unique ID in sequential order.
    If section markers already exist, they are removed first.
    Returns the new content.
    """
    # If markers already exist, remove them.
    if re.search(r"###SECTION:[^#]+###", text) or re.search(r"###ENDSECTION###", text):
        print("Text contains unexpected section markers; removing them")
        text = unwrap_sections(text)
    chunks = split_into_chunks(text, max_chunk_size)
    wrapped_sections = []
    section_counter = 1
    for chunk in chunks:
        section_id = f"section{section_counter}"
        wrapped_sections.append(SECTION_START_FMT.format(section_id=section_id))
        wrapped_sections.append(chunk)
        wrapped_sections.append(SECTION_END)
        section_counter += 1
    return "\n".join(wrapped_sections) + "\n"

def parse_sections(text):
    """
    Parses the text into sections.
    Returns a list of tuples:
        (section_id, section_content, start_line_index, end_line_index)
    and a list of all lines.
    """
    if text is None:
        return ""
    lines = text.splitlines(keepends=True)
    sections = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        start_match = re.match(r"^###SECTION:(\S+)###$", line)
        if start_match:
            section_id = start_match.group(1)
            start_idx = i
            content_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != SECTION_END:
                content_lines.append(lines[i])
                i += 1
            end_idx = i  # Line with the end marker
            section_content = "".join(content_lines).rstrip()
            sections.append((section_id, section_content, start_idx, end_idx))
        i += 1
    return sections, lines

def update_section(text, target_section_id, new_section_text):
    """
    Updates the content of the section with target_section_id.
    If found, replaces the content; if not, appends a new section at the end.
    Returns a tuple (updated_text, updated_flag).
    """
    sections, lines = parse_sections(text)
    for sec in sections:
        section_id, content, start_idx, end_idx = sec
        if section_id == target_section_id:
            # Replace content between start_idx+1 and end_idx (exclusive)
            new_lines = lines[:start_idx+1] + [new_section_text + "\n"] + lines[end_idx:]
            return "".join(new_lines), True
    # If not found, append a new section with a unique ID.
    existing_ids = {sec[0] for sec in sections}
    counter = 1
    while f"section{counter}" in existing_ids:
        counter += 1
    new_section_id = f"section{counter}"
    addition = f"\n{SECTION_START_FMT.format(section_id=new_section_id)}\n{new_section_text}\n{SECTION_END}\n"
    return text + addition, False

def get_section_index(text):
    """
    Returns an index (as an ordered list of section IDs) for the given text.
    """
    sections, _ = parse_sections(text)
    return [sec[0] for sec in sections]

def reorder_sections(text, new_order):
    """
    Reorders the sections in the text based on new_order (a list of section IDs).
    Sections not mentioned in new_order will remain in their original order at the end.
    Returns the updated text.
    """
    sections, lines = parse_sections(text)
    section_map = {sec[0]: (sec[1], sec[2], sec[3]) for sec in sections}
    # Build new content in the order specified.
    new_content = []
    for section_id in new_order:
        if section_id in section_map:
            new_content.append(SECTION_START_FMT.format(section_id=section_id))
            new_content.append(section_map[section_id][0])
            new_content.append(SECTION_END)
    # Append any sections not in new_order.
    remaining = [sec[0] for sec in sections if sec[0] not in new_order]
    for section_id in remaining:
        new_content.append(SECTION_START_FMT.format(section_id=section_id))
        new_content.append(section_map[section_id][0])
        new_content.append(SECTION_END)
    return "\n".join(new_content) + "\n"
