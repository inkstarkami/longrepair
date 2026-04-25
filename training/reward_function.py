# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Custom reward functions for SEARCH/REPLACE Defects4J patch generation.

The total reward is

    score = FORMAT_WEIGHT * format_reward + ANSWER_WEIGHT * answer_reward

where:
- ``format_reward`` is 1 iff the generation contains exactly one matched pair of
  ``<think>...</think>`` and ``<answer>...</answer>`` blocks (in that order)
  and the answer holds at least one well-formed SEARCH/REPLACE block.
- ``answer_reward`` is computed only when ``format_reward`` passes:
    * ``analyze`` task: rewards each generated SEARCH/REPLACE block that
      string-matches a ground-truth patch.
    * ``repair`` task: rewards line-level overlap between the generated diff
      and the ground-truth diff (after stripping comments, blanks and trivial
      duplicates), scaled by a length penalty when the model emits more lines
      than the reference patch. When ``coverage_weight > 0`` the reward is
      blended with a per-hunk coverage signal that fires when the SEARCH side
      contains all removed lines of a hunk.

The two public entry points use the verl reward signature
``(data_source, solution_str, ground_truth, extra_info)``.
"""

import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher

THINK_ANSWER_TAGS = ("<think>", "</think>", "<answer>", "</answer>")
EOS_TOKEN = "<|im_end|>"

SEARCH_REPLACE_PATTERN = re.compile(
    r"^// .*?[\s\S]*?<<<<<<< SEARCH[\s\S]*?=======[\s\S]*?>>>>>>> REPLACE\s*$",
    re.DOTALL,
)
JAVA_BLOCK_PATTERN = re.compile(r"```java(.*?)```", re.DOTALL)
ANY_BLOCK_PATTERN = re.compile(r"```(.*?)```", re.DOTALL)
COMMENT_LINE_PREFIXES = ("//", "*", "/*")

FORMAT_REWARD_WEIGHT = 0.1
ANSWER_REWARD_WEIGHT = 0.9
DEFAULT_COVERAGE_WEIGHT = 0.1
DEFAULT_TARGET_DIFF_LINES = 90
LENGTH_PENALTY_SIGMA_RATIO = 0.84932


def remove_java_line_comments(java_code):
    """Drop Java ``// ...`` line comments while preserving string literals."""
    state_code, state_single, state_double, state_comment = range(4)
    state = state_code
    output_lines = []
    current_line = []
    escape_next = False

    for char in java_code:
        if state == state_comment:
            if char == "\n":
                state = state_code
                current_line.append(char)
                output_lines.append("".join(current_line))
                current_line = []
            continue

        if char == "\n":
            current_line.append(char)
            output_lines.append("".join(current_line))
            current_line = []
            continue

        if state == state_code:
            if current_line and current_line[-1] == "/" and char == "/":
                current_line.pop()
                state = state_comment
            elif char == "'":
                state = state_single
                current_line.append(char)
            elif char == '"':
                state = state_double
                current_line.append(char)
            else:
                current_line.append(char)
        elif state in (state_single, state_double):
            quote = "'" if state == state_single else '"'
            if escape_next:
                current_line.append(char)
                escape_next = False
            elif char == "\\":
                escape_next = True
                current_line.append(char)
            elif char == quote:
                state = state_code
                current_line.append(char)
            else:
                current_line.append(char)

    if current_line:
        output_lines.append("".join(current_line))
    return "".join(output_lines)


def _normalize_diff_lines(diff_text):
    """Strip blanks/comments and return code lines (each ``.strip()``-ed)."""
    cleaned = remove_java_line_comments(diff_text)
    return [
        line.strip()
        for line in cleaned.splitlines()
        if line.strip() and not line.strip().startswith(COMMENT_LINE_PREFIXES)
    ]


def _dedup_diff_pairs(removed, added):
    """Cancel lines that appear on both sides; preserve raw lists if dedup empties.

    Returns ``(removed, added, swap_only)`` where ``swap_only`` is True when the
    dedup left both lists empty (i.e. the diff is a pure reorder).
    """
    raw_removed, raw_added = list(removed), list(added)
    for line in list(removed):
        if line in added:
            removed.remove(line)
            added.remove(line)
    if not removed and not added:
        return raw_removed, raw_added, True
    return removed, added, False


def _contains_all_lines(candidate, required):
    if not required:
        return True
    candidate_counter = Counter(candidate)
    required_counter = Counter(required)
    return all(candidate_counter[line] >= count for line, count in required_counter.items())


def _hunks_from_patch_strings(patches):
    """Group raw SEARCH/REPLACE patch strings into per-file hunks."""
    if patches is None:
        return []
    if not isinstance(patches, list):
        patches = list(patches)
    grouped = {}
    for patch in patches:
        if not isinstance(patch, str):
            continue
        patch = patch.strip()
        if not patch or any(
            marker not in patch
            for marker in ("<<<<<<< SEARCH", "=======", ">>>>>>> REPLACE")
        ):
            continue
        first_line = patch.splitlines()[0].strip()
        file_path = first_line[len("// "):].strip() if first_line.startswith("// ") else first_line
        deleted = patch.split("<<<<<<< SEARCH", 1)[1].split("=======", 1)[0]
        added = patch.split("=======", 1)[1].split(">>>>>>> REPLACE", 1)[0]
        grouped.setdefault(file_path, {"file_path": file_path, "hunks": []})
        grouped[file_path]["hunks"].append(
            {"deleted_lines": deleted, "added_lines": added}
        )
    return list(grouped.values())


def _coverage_hunks(file_hunks):
    """Build per-file hunk records used to track coverage during scoring."""
    coverage = {}
    total = 0
    for file_info in file_hunks:
        file_path = file_info.get("file_path")
        if not file_path:
            continue
        records = []
        for hunk in file_info.get("hunks", []):
            removed, _, _ = _dedup_diff_pairs(
                _normalize_diff_lines(hunk.get("deleted_lines", "")),
                _normalize_diff_lines(hunk.get("added_lines", "")),
            )
            records.append({"deleted_lines": removed, "covered": False})
        coverage[file_path] = records
        total += len(records)
    return coverage, total


def _build_ground_truth_diff(gt_data):
    """Return per-file removed/added lists, swap markers, and total line count."""
    removed_by_file = {}
    added_by_file = {}
    swap_only = {}
    total_lines = 0
    for item in gt_data:
        file_path = item["file_path"]
        removed, added, was_swap = _dedup_diff_pairs(
            _normalize_diff_lines(item["removed_lines"]),
            _normalize_diff_lines(item["added_lines"]),
        )
        if was_swap:
            swap_only[file_path] = True
        removed_by_file[file_path] = removed
        added_by_file[file_path] = added
        total_lines += len(removed) + len(added)
    return removed_by_file, added_by_file, swap_only, total_lines


def _length_penalty(generated_lines, target_lines):
    """Gaussian penalty: 1 when within budget, decays as the model overshoots."""
    if generated_lines <= target_lines:
        return 1.0
    sigma = max(LENGTH_PENALTY_SIGMA_RATIO * target_lines, 1.0)
    return math.exp(-((generated_lines - target_lines) ** 2) / (2 * sigma ** 2))


def _extract_answer_body(solution_str):
    """Return the ``<answer>`` content iff the schema is strictly followed."""
    if solution_str.endswith(EOS_TOKEN):
        solution_str = solution_str[: -len(EOS_TOKEN)]
    solution_str = solution_str.strip()
    if any(solution_str.count(tag) != 1 for tag in THINK_ANSWER_TAGS):
        return None
    open_think, close_think, open_answer, close_answer = (
        solution_str.index(tag) for tag in THINK_ANSWER_TAGS
    )
    if not open_think < close_think < open_answer < close_answer:
        return None
    return solution_str.split("<answer>")[1].split("</answer>")[0].strip()


def _extract_search_replace_blocks(answer_content):
    """Pick out well-formed SEARCH/REPLACE blocks from fenced code regions."""
    if "```java" in answer_content and "```" in answer_content.split("```java")[1]:
        candidates = JAVA_BLOCK_PATTERN.findall(answer_content)
    elif "```" in answer_content and "```" in answer_content.split("```")[1]:
        candidates = ANY_BLOCK_PATTERN.findall(answer_content)
    else:
        return []
    blocks = []
    for block in candidates:
        block = block.strip()
        matches = SEARCH_REPLACE_PATTERN.findall(block)
        if len(matches) == 1:
            blocks.append(matches[0])
    return blocks


def _split_search_replace(search_replace):
    """Return ``(file_path, search_text, replace_text)`` from a SEARCH/REPLACE block."""
    file_path = search_replace.splitlines()[0].replace("// ", "").strip()
    search_text = search_replace.split("<<<<<<< SEARCH")[1].split("=======")[0]
    replace_text = search_replace.split("=======")[1].split(">>>>>>> REPLACE")[0]
    return file_path, search_text, replace_text


def _score_analyze(all_search_replace, extra_info, ground_truth_patches):
    """Reward exact-match SEARCH/REPLACE blocks against ground truth patches."""
    ground_truth_filepath = extra_info.get("filepath", "")
    denom = len(ground_truth_patches) + len(all_search_replace)
    if denom == 0:
        return 0.0
    answer_reward = 0.0
    remaining = list(ground_truth_patches)
    for search_replace in all_search_replace:
        patch_filepath = search_replace.splitlines()[0][len("// "):].strip()
        if patch_filepath not in ground_truth_filepath:
            continue
        for ground_truth_patch in list(remaining):
            if ground_truth_patch == search_replace:
                answer_reward += 0.9 * 2 / denom
                remaining.remove(ground_truth_patch)
                break
    return answer_reward


def _score_repair(all_search_replace, extra_info, ground_truth_patches, coverage_weight):
    """Reward line-level diff overlap, optionally blending per-hunk coverage."""
    ground_truth_filepath = extra_info.get("filepath", "")
    gt_data = extra_info.get("ground_truth", [])
    target_diff_lines = extra_info.get("total_diff_line_count", DEFAULT_TARGET_DIFF_LINES)
    file_hunks = (
        extra_info.get("file_hunks")
        or _hunks_from_patch_strings(ground_truth_patches)
    )

    coverage_hunks, total_hunks = (
        _coverage_hunks(file_hunks) if coverage_weight > 0 else ({}, 0)
    )

    removed_by_file, added_by_file, swap_only, total_gt_lines = (
        _build_ground_truth_diff(gt_data)
    )

    generated_diff_lines = 0
    match_lines = 0
    similarity_scores = {}

    for search_replace in all_search_replace:
        patch_filepath, search_text, replace_text = _split_search_replace(search_replace)
        if patch_filepath not in ground_truth_filepath:
            continue

        generated_diff_lines += (
            len(search_text.strip().splitlines()) + len(replace_text.strip().splitlines())
        )

        search_lines = _normalize_diff_lines(search_text)
        replace_lines = _normalize_diff_lines(replace_text)
        if not swap_only.get(patch_filepath, False):
            for line in list(search_lines):
                if line in replace_lines:
                    search_lines.remove(line)
                    replace_lines.remove(line)

        if coverage_weight > 0:
            for hunk_info in coverage_hunks.get(patch_filepath, []):
                if not hunk_info["covered"] and _contains_all_lines(
                    search_lines, hunk_info["deleted_lines"]
                ):
                    hunk_info["covered"] = True

        if (
            patch_filepath not in removed_by_file
            or patch_filepath not in added_by_file
        ):
            continue

        gt_removed = removed_by_file[patch_filepath]
        gt_added = added_by_file[patch_filepath]
        if not gt_removed:
            found_search_line = True
        else:
            found_search_line = False
            for line in search_lines:
                if line in gt_removed:
                    gt_removed.remove(line)
                    match_lines += 1
                    found_search_line = True
        if not found_search_line:
            continue

        invalid_replace_lines = []
        for line in replace_lines:
            if line in gt_added:
                gt_added.remove(line)
                match_lines += 1
            else:
                invalid_replace_lines.append(line)
        if invalid_replace_lines and gt_added:
            scores = similarity_scores.setdefault(patch_filepath, [])
            for replace_line in invalid_replace_lines:
                best = max(
                    (
                        SequenceMatcher(None, replace_line, gt_line).ratio()
                        for gt_line in gt_added
                    ),
                    default=0.0,
                )
                scores.append(best)

    for patch_filepath, leftover in added_by_file.items():
        scores = similarity_scores.get(patch_filepath, [])
        if not leftover or not scores:
            continue
        if len(scores) > len(leftover):
            scores = sorted(scores, reverse=True)[: len(leftover)]
        match_lines += sum(scores)

    if total_gt_lines == 0:
        total_gt_lines = 1
    match_lines = min(match_lines, total_gt_lines)

    answer_scale = _length_penalty(generated_diff_lines, target_diff_lines)
    line_reward = answer_scale * match_lines / total_gt_lines

    if coverage_weight <= 0:
        return line_reward, None

    if total_hunks > 0:
        covered = sum(
            1 for hunks in coverage_hunks.values() for hunk in hunks if hunk["covered"]
        )
        coverage_reward = answer_scale * covered / total_hunks
    else:
        coverage_reward = line_reward

    blended = (
        (1.0 - coverage_weight) * line_reward + coverage_weight * coverage_reward
    )
    return blended, coverage_reward


def _compute_patch_reward(solution_str, ground_truth, extra_info, coverage_weight):
    extra_info = extra_info or {}
    answer_content = _extract_answer_body(solution_str)

    format_reward = 0.0
    answer_reward = 0.0
    coverage_reward = 0.0

    if answer_content is not None:
        all_search_replace = _extract_search_replace_blocks(answer_content)
        if all_search_replace:
            format_reward = 1.0
            ground_truth_patches = (
                json.loads(ground_truth)
                if isinstance(ground_truth, str)
                else ground_truth
            )
            ground_truth_patches = [p.strip() for p in ground_truth_patches]
            task_type = extra_info.get("task_type", "analyze")
            if task_type == "analyze":
                answer_reward = _score_analyze(
                    all_search_replace, extra_info, ground_truth_patches
                )
            elif task_type == "repair":
                answer_reward, coverage = _score_repair(
                    all_search_replace,
                    extra_info,
                    ground_truth_patches,
                    coverage_weight,
                )
                if coverage is not None:
                    coverage_reward = coverage

    result = {
        "score": FORMAT_REWARD_WEIGHT * format_reward + ANSWER_REWARD_WEIGHT * answer_reward,
        "format_reward": format_reward,
        "answer_reward": answer_reward,
    }
    if coverage_weight > 0:
        result["coverage_reward"] = coverage_reward
    return result


def compute_patch_reward(data_source, solution_str, ground_truth, extra_info=None):
    """verl-compatible reward: format check + line-level diff matching."""
    return _compute_patch_reward(solution_str, ground_truth, extra_info, 0.0)


def compute_patch_reward_with_coverage(
    data_source, solution_str, ground_truth, extra_info=None
):
    """Like ``compute_patch_reward`` but blends in a per-hunk coverage signal."""
    return _compute_patch_reward(
        solution_str, ground_truth, extra_info, DEFAULT_COVERAGE_WEIGHT
    )
