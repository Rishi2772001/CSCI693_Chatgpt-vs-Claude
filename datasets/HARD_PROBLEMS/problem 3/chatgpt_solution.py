class TextEditor:
    def __init__(self):
        # Use two lists to simulate the text editor.
        # 'left' holds the characters to the left of the cursor.
        # 'right' holds the characters to the right of the cursor.
        self.left = []
        self.right = []

    def addText(self, text: str) -> None:
        # Add each character of the text to the left list.
        for char in text:
            self.left.append(char)

    def deleteText(self, k: int) -> int:
        # Delete up to k characters to the left of the cursor.
        count = 0
        while self.left and count < k:
            self.left.pop()
            count += 1
        return count

    def cursorLeft(self, k: int) -> str:
        # Move the cursor to the left by moving characters from left to right.
        count = 0
        while self.left and count < k:
            self.right.append(self.left.pop())
            count += 1
        # Return the last min(10, len(left)) characters to the left of the cursor.
        return ''.join(self.left[-10:])

    def cursorRight(self, k: int) -> str:
        # Move the cursor to the right by moving characters from right to left.
        count = 0
        while self.right and count < k:
            self.left.append(self.right.pop())
            count += 1
        # Return the last min(10, len(left)) characters to the left of the cursor.
        return ''.join(self.left[-10:])
