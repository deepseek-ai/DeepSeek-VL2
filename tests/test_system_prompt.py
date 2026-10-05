import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


def _load_processor():
    # Load the whole processor without importing the model or its GPU dependencies.
    models = Path(__file__).resolve().parents[1] / "deepseek_vl2" / "models"
    package_name = "_vl2_system_prompt_tests"
    package = types.ModuleType(package_name)
    package.__path__ = [str(models)]
    sys.modules[package_name] = package
    name = package_name + ".processing_deepseek_vl_v2"
    spec = importlib.util.spec_from_file_location(name, models / "processing_deepseek_vl_v2.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, importlib.import_module(package_name + ".conversation")


class CharacterTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [ord(character) + 10 for character in text]

    def decode(self, token_ids):
        return "".join(chr(int(token) - 10) for token in token_ids if int(token) >= 10)

    def pad(self, encoded):
        rows = encoded["input_ids"]
        width = max(len(row) for row in rows)
        input_ids = rows[0].new_full((len(rows), width), self.pad_token_id)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        for index, row in enumerate(rows):
            if len(row):
                input_ids[index, -len(row):] = row
                attention_mask[index, -len(row):] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class SystemPromptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module, cls.conversation = _load_processor()

    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.processor = object.__new__(self.module.DeepseekVLV2Processor)
        self.processor.tokenizer = self.tokenizer
        self.processor.sft_format = "deepseek"
        self.processor.mask_prompt = True
        self.processor.ignore_id = -100
        self.processor.image_token = "<image>"
        self.processor.image_token_id = 3
        self.processor.image_size = 1
        self.messages = [
            {"role": "<|User|>", "content": "question"},
            {"role": "<|Assistant|>", "content": "answer"},
        ]

    def _process(self, messages=None, **kwargs):
        return self.processor(
            conversations=self.messages if messages is None else messages,
            images=[],
            force_batchify=False,
            inference_mode=False,
            **kwargs,
        )

    def test_system_prompt_changes_actual_input_ids(self):
        baseline = self._process()
        first = self._process(system_prompt="SYSTEM_ALPHA")
        second = self._process(system_prompt="SYSTEM_BETA")
        self.assertNotEqual(baseline.input_ids.tolist(), first.input_ids.tolist())
        self.assertNotEqual(first.input_ids.tolist(), second.input_ids.tolist())
        self.assertEqual(self.tokenizer.decode(first.input_ids).count("SYSTEM_ALPHA"), 1)
        self.assertEqual(self.tokenizer.decode(second.input_ids).count("SYSTEM_BETA"), 1)

    def test_system_prompt_is_added_only_to_first_turn(self):
        messages = self.messages + [
            {"role": "<|User|>", "content": "second question"},
            {"role": "<|Assistant|>", "content": "second answer"},
        ]
        output = self._process(messages, system_prompt="SYSTEM_ONCE")
        text = self.tokenizer.decode(output.input_ids)
        self.assertEqual(text.count("SYSTEM_ONCE"), 1)
        self.assertTrue(text.startswith("SYSTEM_ONCE"))

    def test_system_prompt_labels_follow_mask_prompt(self):
        tokens = self.tokenizer.encode("SYSTEM_LABELS")
        for masked in (True, False):
            with self.subTest(mask_prompt=masked):
                self.processor.mask_prompt = masked
                output = self._process(system_prompt="SYSTEM_LABELS")
                self.assertEqual(output.input_ids[1:1 + len(tokens)].tolist(), tokens)
                expected = [-100] * len(tokens) if masked else tokens
                self.assertEqual(output.target_ids[1:1 + len(tokens)].tolist(), expected)
                self.assertEqual(output.target_ids[0].item(), self.tokenizer.bos_token_id)

    def test_explicit_system_message_takes_precedence(self):
        for role in ("system", "deepseekapi-sys"):
            with self.subTest(role=role):
                messages = [{"role": role, "content": "EXPLICIT_SYSTEM"}] + self.messages
                first = self._process(messages, system_prompt="IGNORED_ALPHA")
                second = self._process(messages, system_prompt="IGNORED_BETA")
                self.assertEqual(first.input_ids.tolist(), second.input_ids.tolist())
                text = self.tokenizer.decode(first.input_ids)
                self.assertEqual(text.count("EXPLICIT_SYSTEM"), 1)
                self.assertNotIn("IGNORED_ALPHA", text)
                self.assertNotIn("IGNORED_BETA", text)

    def test_empty_system_prompt_preserves_template_defaults(self):
        template = self.conversation.get_conv_template("deepseek")
        template.name = "system_prompt_test_default"
        self.conversation.register_conv_template(template)
        self.addCleanup(self.conversation.conv_templates.pop, template.name)
        self.processor.sft_format = template.name
        for default in ("", "TEMPLATE_DEFAULT"):
            with self.subTest(template_default=default):
                template.set_system_message(default)
                omitted = self._process()
                empty = self._process(system_prompt="")
                direct = self.processor.format_messages_v2(self.messages, [], systems=None)
                self.assertEqual(omitted.input_ids.tolist(), empty.input_ids.tolist())
                self.assertEqual(omitted.target_ids.tolist(), empty.target_ids.tolist())
                self.assertEqual(omitted.input_ids.tolist(), direct[0])
                if default:
                    self.assertEqual(self.tokenizer.decode(omitted.input_ids).count(default), 1)
                override = self._process(system_prompt="OVERRIDE_SYSTEM")
                text = self.tokenizer.decode(override.input_ids)
                self.assertTrue(text.startswith("OVERRIDE_SYSTEM"))
                self.assertEqual(text.count("OVERRIDE_SYSTEM"), 1)
                self.assertNotIn("TEMPLATE_DEFAULT", text)

    def test_batched_call_preserves_system_ids_and_labels(self):
        unbatched = self._process(system_prompt="BATCH_SYSTEM")
        batched = self.processor(
            conversations=self.messages,
            images=[],
            force_batchify=True,
            system_prompt="BATCH_SYSTEM",
        )
        self.assertEqual(batched.input_ids[0].tolist(), unbatched.input_ids[:-1].tolist())
        self.assertEqual(batched.labels[0].tolist(), unbatched.target_ids[:-1].tolist())
        self.assertEqual(self.tokenizer.decode(batched.input_ids[0]).count("BATCH_SYSTEM"), 1)
        self.assertTrue(batched.attention_mask.all().item())


if __name__ == "__main__":
    unittest.main()
