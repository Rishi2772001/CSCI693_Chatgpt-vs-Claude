class TextEditor:
    def __init__(self):
        """
        Initialize the TextEditor with empty text and cursor at position 0.
        """
        self.text = ""
        self.cursor_position = 0
    
    def addText(self, text: str) -> None:
        """
        Appends text at the current cursor position.
        The cursor ends to the right of the added text.
        
        Args:
            text: The string to append at the cursor position
        """
        # Insert the text at the cursor position
        self.text = self.text[:self.cursor_position] + text + self.text[self.cursor_position:]
        # Move the cursor to the end of the inserted text
        self.cursor_position += len(text)
    
    def deleteText(self, k: int) -> int:
        """
        Deletes k characters to the left of the cursor.
        
        Args:
            k: Number of characters to delete
            
        Returns:
            The number of characters actually deleted
        """
        # Calculate how many characters we can actually delete
        chars_to_delete = min(k, self.cursor_position)
        
        # Delete the characters
        self.text = self.text[:self.cursor_position - chars_to_delete] + self.text[self.cursor_position:]
        # Update the cursor position
        self.cursor_position -= chars_to_delete
        
        return chars_to_delete
    
    def cursorLeft(self, k: int) -> str:
        """
        Moves the cursor to the left k times.
        
        Args:
            k: Number of positions to move the cursor left
            
        Returns:
            The last min(10, len) characters to the left of the cursor
        """
        # Calculate the new cursor position, ensuring it doesn't go below 0
        self.cursor_position = max(0, self.cursor_position - k)
        
        # Return the last min(10, len) characters to the left of the cursor
        start = max(0, self.cursor_position - 10)
        return self.text[start:self.cursor_position]
    
    def cursorRight(self, k: int) -> str:
        """
        Moves the cursor to the right k times.
        
        Args:
            k: Number of positions to move the cursor right
            
        Returns:
            The last min(10, len) characters to the left of the cursor
        """
        # Calculate the new cursor position, ensuring it doesn't exceed the text length
        self.cursor_position = min(len(self.text), self.cursor_position + k)
        
        # Return the last min(10, len) characters to the left of the cursor
        start = max(0, self.cursor_position - 10)
        return self.text[start:self.cursor_position]