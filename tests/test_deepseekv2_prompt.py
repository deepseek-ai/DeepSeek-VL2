import importlib.util
import unittest
from pathlib import Path


def _load_conversation():
    path = Path(__file__).resolve().parents[1] / "deepseek_vl2" / "models" / "conversation.py"
    spec = importlib.util.spec_from_file_location("deepseek_vl2_conversation_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeepseekV2PromptTest(unittest.TestCase):
    def test_a_user_turn_is_not_stored_as_an_assistant_turn(self):
        conv = _load_conversation().get_conv_template("deepseekv2")
        conv.append_message(conv.roles[0], "Hello!")
        conv.append_message(conv.roles[1], "Hi!")
        prompt = conv.get_prompt()
        self.assertEqual(
            prompt,
            "<｜sft▁begin｜>\nHello!" + conv.sep + "Hi!" + conv.sep2,
        )

    def test_the_v1_template_still_names_the_user(self):
        conv = _load_conversation().get_conv_template("deepseek")
        conv.append_message(conv.roles[0], "Hello!")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        self.assertTrue(prompt.startswith("<|User|>: Hello!"))
        self.assertTrue(prompt.endswith("<|Assistant|>:"))


if __name__ == "__main__":
    unittest.main()
