"""A tiny hand-written corpus whose answers can be checked by eye."""
from asofqa.index import Corpus, parse_doc


def doc(doc_id, version, status, effective, body, policy="pto", title="Paid Time Off Policy"):
    return parse_doc(
        f"---\ndoc_id: {doc_id}\npolicy: {policy}\ntitle: {title}\nversion: {version}\n"
        f"status: {status}\neffective: {effective}\n---\n# {title}\n\n{body}\n"
    )


PTO = [
    doc("pto-v1", 1, "active", "2021-01-01",
        "Full-time employees receive 20 days of paid time off per calendar year. "
        "Requests should be submitted at least 5 days in advance."),
    doc("pto-v2", 2, "withdrawn", "2022-03-01",
        "Full-time employees receive 30 days of paid time off per calendar year.\n\n"
        "WITHDRAWN. This proposal was withdrawn on 2022-02-20 and never took effect."),
    doc("pto-v3", 3, "active", "2023-07-01",
        "Requests should be submitted at least 10 days in advance. "
        "Full-time employees receive 25 days of paid time off per calendar year.\n\n"
        "This version replaces the previous figure of 20 days, in effect since 2021-01-01."),
    doc("pto-v4", 4, "active", "2027-01-01",
        "Full-time employees receive 27 days of paid time off per calendar year.\n\n"
        "This version replaces the previous figure of 25 days, in effect since 2023-07-01."),
    doc("pto-v5", 5, "proposed", "2027-06-01",
        "Full-time employees receive 35 days of paid time off per calendar year.\n\n"
        "DRAFT. This proposal is under review and has not been approved."),
]
MEAL = [
    doc("meal-v1", 1, "active", "2020-06-01", "The daily meal allowance during business travel is $50.",
        policy="meal", title="Business Travel Meal Policy"),
    doc("meal-v2", 2, "active", "2024-02-01",
        "The daily meal allowance during business travel is $65. Receipts must be submitted within 30 days.",
        policy="meal", title="Business Travel Meal Policy"),
]


def mini_corpus() -> Corpus:
    return Corpus(PTO + MEAL)
