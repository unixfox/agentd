# TODO: use or delte
class SectionBasedPrompts():
    def __init__(self):
        super().__init__()
        self.instructions = (
            "You are using the 'Section-Based' edit format. The source file is divided into sections "
            "using the following markers:\n"
            "  - Section start marker: \"```file.sectionID\" (e.g., ```file1.section1)\n"
            "  - Section end marker: \"```\"\n\n"
            "When making code changes, return your edits as a complete, updated code block. "
            "If you include a section label (for example, 'file1.section1') in the opening backticks, "
            "only that section will be updated. If you do not provide a section label (i.e. the opening backticks are empty), "
            "assume that the entire file should be replaced with your output. For example:\n\n"
            "To update a section:\n"
            "```file1.section1\n<updated code>\n```\n\n"
            "To replace the entire file:\n"
            "```\n<new full file content>\n```\n\n"
            "By default, if a new section is created, it will be appended at the end."
        )

    def get_prompt(self, file_content, additional_instructions):
        base_prompt = super().get_prompt(file_content, additional_instructions)
        return base_prompt + "\n" + self.instructions
