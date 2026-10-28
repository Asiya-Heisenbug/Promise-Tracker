import unittest
from unittest.mock import Mock, patch

import analyzer
import nlp
import scraper


class JobAnalysisTests(unittest.TestCase):
    def test_markdown_job_description_heading_is_cleaned(self):
        text = "# Job Description: Sales Management Executive\n\nCompany: Example Corp"
        job = nlp.extract_job(text)
        self.assertEqual(job["title"], "Sales Management Executive")
        self.assertEqual(job["company"], "Example Corp")

    def test_job_title_is_not_misidentified_as_company(self):
        text = "# Job Description: Sales Management Executive\n"
        self.assertIsNone(nlp.extract_company(text))

    def test_sales_responsibilities_describe_sales_work(self):
        job = {
            "responsibilities": [
                "Generate leads and meet monthly sales targets.",
                "Build customer relationships and manage the sales pipeline.",
            ],
            "raw_text": "Requirements: machine learning and model training.",
        }
        result = analyzer.plain_language(job)
        self.assertIn("sales targets", result["summary"])
        self.assertIn("potential customers", result["summary"])
        self.assertNotIn("machine-learning models", result["summary"])

    def test_extracted_sales_duties_are_used_for_plain_language_summary(self):
        text = (
            "# Job Description: Sales Management Executive\n"
            "Company: Example Corp\n\n"
            "Responsibilities\n"
            "- Generate leads and meet monthly sales targets.\n"
            "- Build customer relationships and manage the sales pipeline.\n\n"
            "Requirements\n"
            "- Knowledge of machine learning and model training.\n"
        )
        job = nlp.extract_job(text)
        result = analyzer.plain_language(job)
        self.assertEqual(job["title"], "Sales Management Executive")
        self.assertEqual(job["company"], "Example Corp")
        self.assertIn("sales targets", result["summary"])
        self.assertNotIn("machine-learning models", result["summary"])

    def test_unrelated_job_text_is_not_used_as_day_to_day_duties(self):
        job = {
            "responsibilities": [],
            "raw_text": "Title: Sales Executive. Requirements: machine learning and model training.",
        }
        result = analyzer.plain_language(job)
        self.assertIn("couldn't find enough information", result["summary"])
        self.assertEqual(result["bullets"], [])


class CompanyLookupTests(unittest.TestCase):
    @patch("scraper.requests.get")
    def test_wikipedia_search_resolves_a_company_article(self, get):
        search = Mock(status_code=200)
        search.json.return_value = {
            "query": {"search": [{"title": "Acme Robotics", "pageid": 123}]}
        }
        summary = Mock(status_code=200)
        summary.json.return_value = {
            "type": "standard",
            "title": "Acme Robotics",
            "extract": "Acme Robotics makes industrial automation products.",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Acme_Robotics"}},
        }
        get.side_effect = [search, summary]

        self.assertEqual(
            scraper._wikipedia("Acme Robotics"),
            (
                "Acme Robotics makes industrial automation products.",
                "https://en.wikipedia.org/wiki/Acme_Robotics",
            ),
        )
        self.assertEqual(get.call_count, 2)
        self.assertIn("w/api.php", get.call_args_list[0].args[0])

    @patch("scraper.requests.get")
    def test_wikipedia_search_ignores_unrelated_results(self, get):
        search = Mock(status_code=200)
        search.json.return_value = {
            "query": {"search": [{"title": "Sales management"}]}
        }
        get.return_value = search
        self.assertIsNone(scraper._wikipedia("Acme Robotics"))
        self.assertEqual(get.call_count, 1)


if __name__ == "__main__":
    unittest.main()
