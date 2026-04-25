"""Generate SEARCH/REPLACE patches for Defects4J bugs with a causal LLM."""

import json
import os

import fire
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompter import (
    D4J_SYSTEM_PROMPT,
    D4J_USER_PROMPT_FUNCTION_ONLY,
    D4J_USER_PROMPT_WITH_SKELETON,
)

PAD_TOKEN_ID = 151643


def load_model_and_tokenizer(
    base_model_path,
    tokenizer_path,
    use_lora=False,
    lora_weights=None,
    use_new_tokens=False,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token_id = PAD_TOKEN_ID
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if use_new_tokens:
        model.resize_token_embeddings(len(tokenizer))

    if use_lora and lora_weights:
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )

    return model, tokenizer


def build_problem_statement(data, have_summary):
    if have_summary:
        return "summary: " + data["summary"] + "\ndescription: " + data["description"]
    return "summary: UNKNOWN\ndescription: UNKNOWN"


def build_buggy_content(data, have_skeleton):
    content = ""
    for file in data["files"]:
        source_path = file["source_path"]
        file_skeleton = file["processed_source"]
        functions = file["functions"]
        context_blocks = []
        for function in functions:
            source_context = function["source_context"]
            if isinstance(source_context, list):
                context_blocks.append("\n".join(source_context))
            else:
                context_blocks.append(source_context)
        joined_context = ("\n\n".join(context_blocks)).strip()
        if have_skeleton:
            content += (
                "// "
                + source_path
                + "\n// buggy functions: \n"
                + joined_context
                + "\n\n// file skeleton: \n\n"
                + file_skeleton
                + "\n\n"
            )
        else:
            content += (
                "// "
                + source_path
                + "\n// buggy functions: \n"
                + joined_context
                + "\n\n"
            )
    return content


def load_processed_bugs(output_path):
    if not os.path.exists(output_path):
        return []
    processed_bugs = []
    valid_lines = []
    with open(output_path, "r") as f_out:
        for line in f_out.readlines():
            data = json.loads(line.strip())
            if data["output"]:
                processed_bugs.append(data["bug_id"])
                valid_lines.append(line)
    with open(output_path, "w") as f_out:
        for line in valid_lines:
            f_out.write(line)
    return processed_bugs


def generate_defects4j_patches(
    dataset_path: str = "../dataset/defects4j_benchmark.jsonl",
    output_path: str = "../results/generation_outputs/run.jsonl",
    base_model: str = "",
    use_lora: bool = False,
    use_new_tokens: bool = False,
    lora_weights: str = ".",
    max_new_tokens: int = 2048,
    num_beams: int = 3,
    n: int = 1,
    start_index: int = 0,
    end_index: int = -1,
    have_summary: bool = False,
    have_skeleton: bool = False,
):
    if not base_model:
        raise ValueError("`base_model` is required.")

    tokenizer_path = lora_weights if (use_lora and use_new_tokens) else base_model
    model, tokenizer = load_model_and_tokenizer(
        base_model,
        tokenizer_path,
        use_lora=use_lora,
        lora_weights=lora_weights,
        use_new_tokens=use_new_tokens,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_bugs = load_processed_bugs(output_path)

    user_template = (
        D4J_USER_PROMPT_WITH_SKELETON if have_skeleton else D4J_USER_PROMPT_FUNCTION_ONLY
    )

    with open(dataset_path, "r") as f_in, open(output_path, "a") as f_out:
        for index, line in enumerate(tqdm(f_in)):
            if index < start_index or 0 <= end_index <= index:
                continue
            data = json.loads(line.strip())
            bug_id = data["bug_id"]
            if bug_id in processed_bugs:
                continue

            problem_statement = build_problem_statement(data, have_summary)
            content = build_buggy_content(data, have_skeleton)

            full_prompt = user_template.format(
                problem_statement=problem_statement,
                content=content.strip(),
            )
            messages = [
                {"role": "system", "content": D4J_SYSTEM_PROMPT},
                {"role": "user", "content": full_prompt},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(
                text,
                truncation=True,
                padding=False,
                return_tensors="pt",
            )
            eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

            patches = []
            try:
                outputs = model.generate(
                    input_ids=inputs["input_ids"].cuda(),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_id,
                    max_new_tokens=max_new_tokens,
                    temperature=0.6,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    num_return_sequences=n,
                    do_sample=True,
                )
            except Exception as exc:
                print(f"Generation failed for bug {bug_id}: {exc}")
            else:
                for output in outputs:
                    raw_patch = tokenizer.decode(
                        output[len(inputs[0]):],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    raw_patch_split = raw_patch.split("<|endoftext|>")
                    patch_tokens = (
                        len(output) - len(inputs[0]) - (len(raw_patch_split) - 1)
                    )
                    patches.append(
                        {"raw_patch": raw_patch_split[0], "tokens": patch_tokens}
                    )

            result = {
                "bug_id": bug_id,
                "input": text,
                "output": patches,
            }
            f_out.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    fire.Fire(generate_defects4j_patches)
