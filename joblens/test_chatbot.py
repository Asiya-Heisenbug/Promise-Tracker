import unittest
from unittest.mock import patch

import chatbot


def make_context():
    checks = [
        {"name": "Salary", "score": None, "note": "You didn't set a minimum salary."},
        {"name": "Location", "score": None, "note": "Location is not clearly mentioned."},
        {"name": "Shift", "score": None, "note": "You have no shift preference."},
        {"name": "Experience", "score": None, "note": "Required experience is not mentioned."},
    ]
    return {
        "job": {
            "title": "AI Developer",
            "salary": {"text": "₹40,000 per month"},
            "location": "Chennai",
            "work_mode": "Hybrid",
            "shift": "Morning/Day",
            "experience": {"text": "Fresher / entry level"},
            "required_skills": ["Python", "machine learning"],
            "preferred_skills": ["FastAPI"],
            "responsibilities": ["Build and test machine-learning models."],
            "raw_text": "private full job description",
        },
        "fit": {
            "percent": 80,
            "label": "Strong match",
            "explanation": "**Why it fits:** Good role match.",
            "checks": checks,
            "skills": {
                "matched": ["Python"],
                "related": {},
                "missing_required": ["machine learning"],
                "missing_preferred": ["FastAPI"],
            },
            "criteria_used": "2 of 6",
        },
        "plain": {
            "summary": "You'll mainly spend your time building models.",
            "bullets": ["Building and evaluating machine-learning models"],
        },
        "company": None,
        "reviews": None,
    }


class ChatbotTests(unittest.TestCase):
    def setUp(self):
        self.context = make_context()

    def test_answers_from_job_fields_even_when_fit_check_is_unavailable(self):
        self.assertIn("₹40,000 per month", chatbot.answer("What's the salary?", self.context))
        self.assertIn("Chennai", chatbot.answer("Where is the company?", self.context))
        self.assertIn("Morning/Day", chatbot.answer("What shift is this?", self.context))
        self.assertIn("Fresher / entry level", chatbot.answer("How much experience?", self.context))

    def test_work_mode_is_returned_when_city_is_missing(self):
        self.context["job"]["location"] = None
        self.assertIn("Hybrid", chatbot.answer("Is this remote or hybrid?", self.context))

    def test_fit_with_too_few_criteria_shows_partial_explanation(self):
        self.context["fit"]["percent"] = None
        reply = chatbot.answer("How suitable is this role?", self.context)
        self.assertIn("fewer than three criteria", reply)
        self.assertIn("Good role match", reply)

    def test_apply_advice_is_not_invented_without_fit_score(self):
        self.context["fit"]["percent"] = None
        reply = chatbot.answer("Should I apply?", self.context)
        self.assertIn("can't give a reliable apply recommendation", reply)
        self.assertNotIn("It looks worth applying", reply)

    def test_skills_and_help_are_available_without_llm(self):
        self.assertIn("machine learning", chatbot.answer("What skills am I missing?", self.context))
        self.assertIn("salary, location, shift", chatbot.answer("help", self.context))

    def test_empty_question_is_handled(self):
        self.assertIn("enter a question", chatbot.answer("  ", self.context))

    def test_llm_fallback_receives_structured_context_without_raw_job_text(self):
        with patch.object(chatbot, "ask_llm", return_value="Grounded answer") as ask_llm:
            self.assertEqual(chatbot.answer("Does it include mentoring?", self.context), "Grounded answer")
        prompt = ask_llm.call_args.args[0]
        self.assertNotIn("private full job description", prompt)
        self.assertIn("machine learning", prompt)


if __name__ == "__main__":
    unittest.main()
