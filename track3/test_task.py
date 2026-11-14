"""Checks for easy ways to cheat the arithmetic reward."""

import unittest

from task import check_answer


TASK = {"numbers": [3, 7, 25], "target": 46}


class CheckAnswerTests(unittest.TestCase):
    def test_correct_expression(self):
        self.assertTrue(check_answer(TASK, "<answer>25+3*7</answer>")["correct"])

    def test_repeated_or_missing_number(self):
        self.assertFalse(check_answer(TASK, "<answer>25+7+7+7</answer>")["correct"])
        self.assertFalse(check_answer(TASK, "<answer>46</answer>")["correct"])

    def test_extra_text_and_multiple_answers(self):
        self.assertFalse(check_answer(TASK, "<answer>25+3*7</answer> or 46")["correct"])
        self.assertFalse(check_answer(TASK, "<answer>25+3*7</answer><answer>46</answer>")["correct"])

    def test_reject_code_and_division_by_zero(self):
        self.assertFalse(check_answer(TASK, '<answer>__import__("os")</answer>')["correct"])
        self.assertFalse(check_answer(TASK, "<answer>25+3/(7-7)</answer>")["correct"])

    def test_exact_fraction(self):
        task = {"numbers": [2, 3, 6], "target": 9}
        self.assertTrue(check_answer(task, "<answer>6/2*3</answer>")["correct"])
        self.assertFalse(check_answer(task, "<answer>6/3*2</answer>")["correct"])


if __name__ == "__main__":
    unittest.main()
