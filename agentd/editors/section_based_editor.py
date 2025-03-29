import re

from agentd.section_chunker import update_section


class SectionBasedEditor():
    @staticmethod
    def apply_edits(file_content, edits):
        """
        Parses the LLM's edit response and applies the edits.
        If multiple sections are specified, each section is updated.
        If no section label is provided, the entire file is replaced.
        """
        pattern = r"```(\S*)\n(.*?)\n```"
        matches = re.findall(pattern, edits, re.DOTALL)

        if not matches:
            raise ValueError("Edit response did not match the expected format.")

        updated_content = file_content

        for target, updated_code in matches:
            target = target.strip()  # May be empty

            if target == "":
                # If any edit block has no target, replace the full file.
                updated_content = updated_code
                break  # No need to process further since the whole file is replaced.

            else:
                # Update individual sections
                updated_content, _ = update_section(updated_content, target, updated_code)

        return updated_content
