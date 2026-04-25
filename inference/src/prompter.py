"""Prompt templates used by the Defects4J generation pipeline."""

D4J_SYSTEM_PROMPT = """
You are a helpful bug fixer that provides well-reasoned and detailed bug fixes.
We are currently solving the following issue within our repository. Please generate ```java``` code snippet containing *SEARCH/REPLACE* edits to fix the issue.

Each *SEARCH/REPLACE* edit must wrap in ```java``` code snippet and use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```java
// com/example/myapp/Main.java
<<<<<<< SEARCH
        System.out.println("Hello World");
=======
        System.out.println("Hello World");
        ArrayList<String> list = new ArrayList<>();
>>>>>>> REPLACE
```

Ensure the *SEARCH* block is contiguous and minimal, should only contain buggy lines and necessary surrounding lines, and should not contain the full method if not necessary.
We are given three parts of the issue information:
    - Issue text, which contains summary and description of the issue.
    - Buggy code segment.
    - New import files(if any), which may be used in the fix code.
"""

D4J_USER_PROMPT_WITH_SKELETON = """
Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Below are some code segments, each from a relevant file.
The code segment is provided in a specific format: for each file, the file path is given first, followed by the code of the functions relevant to the bug, and finally the skeleton of the file (e.g., class declarations, imports, etc.).
--- BEGIN FILE ---
```
{content}
```
--- END FILE ---
"""

D4J_USER_PROMPT_FUNCTION_ONLY = """
Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Below are some code segments, each from a relevant file.
The code segment is provided in a specific format: for each file, the file path is given first, followed by the code of the functions relevant to the bug in the file.
--- BEGIN FILE ---
```
{content}
```
--- END FILE ---
"""
