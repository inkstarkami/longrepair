"""Validate generated SEARCH/REPLACE patches against Defects4J bugs."""

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import javalang

os.environ['TZ'] = 'America/Los_Angeles'
tmp_folder = "/tmp/"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILENAME_FILE = REPO_ROOT / "dataset" / "defects4j_filename_map.json"
MAX_PATCH_TOKENS = 4096
MAX_PATCHES_PER_BUG = 10


def extract_search_replace_blocks(text):
    # Step 1: Extract answer content
    if '<answer>' in text and '</answer>' in text:
        start_idx = text.find('<answer>') + len('<answer>')
        end_idx = text.find('</answer>', start_idx)
        if end_idx != -1:
            content = text[start_idx:end_idx]
        else:
            content = text[start_idx:]
    elif '<answer>' in text:
        start_idx = text.find('<answer>') + len('<answer>')
        content = text[start_idx:]
    elif '</think>' in text:
        start_idx = text.find('</think>') + len('</think>')
        content = text[start_idx:]
    else:
        content = text

    blocks = []

    # Step 2: Extract from markdown code blocks
    markdown_blocks = re.findall(r'```(?:java)?\s*([\s\S]*?)```', content)
    for block in markdown_blocks:
        sr_blocks = re.findall(r'.*\.java\n<<<<<<< SEARCH[\s\S]*?>>>>>>> REPLACE', block)
        for sr_block in sr_blocks:
            if not sr_block.startswith('// '):
                if '// ' in sr_block.splitlines()[0]:
                    sr_block = '// ' + sr_block.split('// ')[1]
                else:
                    sr_block = '// ' + sr_block
            if sr_block.startswith('// File: '):
                sr_block = '// ' + sr_block[len('// File: '):]
            if "// Buggy line" in sr_block:
                sr_block = sr_block.replace("// Buggy line", "")
            blocks.append(sr_block)

    # Step 3: Extract from raw text if no blocks found and SEARCH exists
    if not blocks and '<<<<<<< SEARCH' in text:
        raw_blocks = re.findall(r'.*\.java\n<<<<<<< SEARCH[\s\S]*?>>>>>>> REPLACE', text)
        for block in raw_blocks:
            if not block.startswith('// '):
                block = '// ' + block
            if block.startswith('// File: '):
                block = '// ' + block[len('// File: '):]
            if "// Buggy line" in block:
                block = block.replace("// Buggy line", "")
            blocks.append(block)

    return blocks


def process_patch_file(input_path):
    """Read generation outputs and keep up to MAX_PATCHES_PER_BUG in-budget patches per bug.

    Generations whose decoded length is at or above ``MAX_PATCH_TOKENS`` are
    discarded; they are unlikely to be well-formed SEARCH/REPLACE blocks.
    """
    bug_patches = []
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line)
            bug_id = data["bug_id"]
            patches = []
            for item in data.get("output", []):
                if item.get("tokens", MAX_PATCH_TOKENS) >= MAX_PATCH_TOKENS:
                    continue
                raw_patch = item.get("raw_patch", "")
                blocks = extract_search_replace_blocks(raw_patch) if raw_patch else []
                patches.append(blocks)
                if len(patches) >= MAX_PATCHES_PER_BUG:
                    break
            bug_patches.append({"bug_id": bug_id, "output": patches})
    return bug_patches


def run_d4j_test(sources, testmethods, bug_id):
    bugg = False
    compile_fail = False
    timed_out = False
    entire_bugg = False
    error_string = ""

    for source in sources:
        try:
            tokens = javalang.tokenizer.tokenize(source)
            parser = javalang.parser.Parser(tokens)
            parser.parse()
        except:
            print("Syntax Error")
            return compile_fail, timed_out, bugg, entire_bugg, True, None

    for t in testmethods:
        print(t.strip())
        cmd = 'defects4j test -w %s/ -t %s' % ((tmp_folder + bug_id), t.strip())
        Returncode = ""
        stderr_path = Path(tmp_folder) / f"{bug_id}_stderr.txt"
        error_file = open(stderr_path, "wb")
        child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=error_file, bufsize=-1,
                                 start_new_session=True)
        while_begin = time.time()
        while True:
            Flag = child.poll()
            if Flag == 0:
                Returncode = child.stdout.readlines()  # child.stdout.read()
                print(b"".join(Returncode).decode('utf-8'))
                error_file.close()
                break
            elif Flag != 0 and Flag is not None:
                compile_fail = True
                error_file.close()
                with open(stderr_path, "rb") as f:
                    r = f.readlines()
                for line in r:
                    if re.search(r':\serror:\s', line.decode('utf-8')):
                        error_string = line.decode('utf-8')
                        break
                print(error_string)
                break
            elif time.time() - while_begin > 15:
                error_file.close()
                os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                timed_out = True
                break
            else:
                time.sleep(0.01)
        log = Returncode
        if len(log) > 0 and log[-1].decode('utf-8') == "Failing tests: 0\n":
            continue
        else:
            bugg = True
            break

    # Then we check if it passes all the tests, include the previously okay tests
    if not bugg:
        print('So you pass the basic tests, Check if it passes all the test, include the previously passing tests')
        # cmd = 'defects4j test -w %s/' % (tmp_folder + bug_id)
        cmd = 'defects4j test -r -w %s/' % (tmp_folder + bug_id)
        Returncode = ""
        child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1,
                                 start_new_session=True)
        while_begin = time.time()
        while True:
            Flag = child.poll()
            if Flag == 0:
                Returncode = child.stdout.readlines()  # child.stdout.read()
                break
            elif Flag != 0 and Flag is not None:
                bugg = True
                compile_fail = True
                break
            elif time.time() - while_begin > 180:
                os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                bugg = True
                timed_out = True
                break
            else:
                time.sleep(0.01)
        log = Returncode
        if len(log) > 0 and log[-1].decode('utf-8') == "Failing tests: 0\n":
            print('success')
        else:
            entire_bugg = True

    return compile_fail, timed_out, bugg, entire_bugg, False, log


def extract_search_replace(snippets):
    patches = []
    for snippet in snippets:
        lines = snippet.splitlines()
        if not lines:
            return patches

        filename = lines[0].lstrip('/').strip()

        if len(lines) < 4 or lines[1] != "<<<<<<< SEARCH":
            # raise ValueError("Invalid SEARCH/REPLACE block: missing SEARCH marker")
            patches.append([filename, "", ""])
            continue

        try:
            divider_idx = lines.index("=======", 2)
            replace_end_idx = lines.index(">>>>>>> REPLACE", divider_idx + 1)
        except ValueError:
            patches.append([filename, "", ""])
            continue
            # raise ValueError("Invalid SEARCH/REPLACE block: missing divider or REPLACE marker")

        search_lines = lines[2:divider_idx]
        search_content = '\n'.join(search_lines)

        replace_lines = lines[divider_idx + 1:replace_end_idx]
        replace_content = '\n'.join(replace_lines)

        patch = [filename, search_content, replace_content]
        patches.append(patch)
    return patches


def add_bug_comment(source, bug_location):
    print(bug_location)
    if not source:
        return source
    source_lines = source.splitlines()
    for bug_line in bug_location:
        bug_line -= 1
        try:
            source_lines[bug_line] += " // Buggy Line"
        except IndexError:
            print("")
        try:
            print(source_lines[bug_line - 1])
            print(source_lines[bug_line])
            print(source_lines[bug_line + 1])
        except IndexError:
            if bug_line < len(source_lines):
                print(source_lines[bug_line])
    edited_source = "\n".join(source_lines)
    return edited_source


def validate_patch(bug_id, patch, filenames):
    project = bug_id.split("-")[0]
    bug = bug_id.split("-")[1]
    tmp_bug_id = "test_" + bug_id

    if not patch:
        print("skip test")
        return False, ""

    print("defects4j checkout -p %s -v %s -w %s" % (project, bug + 'b', (tmp_folder + tmp_bug_id)))

    subprocess.run('rm -rf ' + tmp_folder + tmp_bug_id, shell=True)
    subprocess.run("defects4j checkout -p %s -v %s -w %s" % (project, bug + 'b', (tmp_folder + tmp_bug_id)), shell=True)
    testmethods = os.popen('defects4j export -w %s -p tests.trigger' % (tmp_folder + tmp_bug_id)).readlines()

    patch.reverse()
    replaces = extract_search_replace(patch)
    sources = []
    for replace in replaces:
        source_path = replace[0]
        search_content = replace[1]
        replace_content = replace[2]
        source_filename = source_path.split('/')[-1]
        source_filename_data = filenames.get(source_filename, {})
        print(source_filename_data.get('source_path', "None"))
        source_path = source_filename_data.get('source_path', source_path)
        bug_location = source_filename_data.get('bug_location', [])
        print(tmp_folder + tmp_bug_id + "/" + source_path)
        try:
            with open(tmp_folder + tmp_bug_id + "/" + source_path) as f:
                lines = f.readlines()
                source = "\n".join([line.strip() for line in lines])
        except:
            try:
                with open(tmp_folder + tmp_bug_id + "/" + source_path, encoding='ISO-8859-1') as f:
                    lines = f.readlines()
                    source = "\n".join([line.strip() for line in lines])
            except FileNotFoundError:
                source = ""
                print("wrong filename")
        search_content = "\n" + "\n".join([line.strip() for line in search_content.splitlines()])
        if search_content.strip():
            if search_content in source:
                print("matched source")
                source = source.replace(search_content, replace_content)
                sources.append(source)
            else:
                if "// Buggy Line" in search_content:
                    edited_source = add_bug_comment(source, bug_location)
                    if search_content in edited_source:
                        print("matched with bug comment")
                        edited_source = edited_source.replace(search_content, replace_content)
                        source = edited_source
                    sources.append(edited_source)

        try:
            with open(tmp_folder + tmp_bug_id + "/" + source_path, "w") as f:
                f.write(source)
        except:
            try:
                with open(tmp_folder + tmp_bug_id + "/" + source_path, "w", encoding='ISO-8859-1') as f:
                    f.write(source)
            except:
                print("")

    compile_fail, timed_out, bugg, entire_bugg, syntax_error, log = run_d4j_test(sources, testmethods, tmp_bug_id)

    subprocess.run('rm -rf ' + tmp_folder + tmp_bug_id, shell=True)

    if not compile_fail and not timed_out and not bugg and not entire_bugg and not syntax_error:
        print("{} has valid patch: {{}}".format(bug_id))
        return True, None
    print("{} has invalid patch: {{}}".format(bug_id))
    if compile_fail:
        message = "Compile Fail"
    elif timed_out:
        message = "Time Out"
    elif syntax_error:
        message = "Syntax Error"
    else:
        message = "Failing test: " + log[-1].decode('utf-8')[4:-1]
    return False, message


def main(input_file, filename_file):
    with open(filename_file, 'r') as f_filename:
        converted_filenames = json.load(f_filename)

    result_file = input_file.replace(".jsonl", "_result.jsonl")
    tested = set()
    if os.path.exists(result_file):
        with open(result_file, 'r') as f_out:
            for line in f_out:
                tested.add(json.loads(line)['bug_id'])

    for entry in process_patch_file(input_file):
        bug_id = entry['bug_id']
        if bug_id in tested:
            continue
        passed = False
        results = []
        filenames = converted_filenames.get(bug_id, {})
        for patch in entry['output']:
            correct, _ = validate_patch(bug_id, patch, filenames)
            if correct:
                passed = True
            results.append(correct)
        with open(result_file, 'a+') as f_out:
            f_out.write(
                json.dumps({'bug_id': bug_id, 'passed': passed, 'results': results}) + '\n'
            )


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python validate_patch.py <input_file>")
    input_file = sys.argv[1].strip()
    filename_file = os.environ.get("FILENAME_FILE", str(DEFAULT_FILENAME_FILE))
    tmp_folder = "/tmp/" + Path(input_file).stem + "/"
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
    main(input_file, filename_file)
