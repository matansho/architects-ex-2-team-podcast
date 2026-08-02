"""Lightweight tests for verify v2 (no LLM). Run: .venv/bin/python -m rag.test_verify"""

from __future__ import annotations

from rag.verify import (
    covers_question,
    filter_protected_spans,
    should_verify,
    strip_unasked_asides,
    strip_unsupported,
)


def test_strip_extras_41_43_46() -> None:
    cases = [
        (
            "dev-41",
            "אילו ביטוחי משכנתא בכלל יש אצלכם בהראל, ואם אני רוצה את ביטוח החיים למשכנתא - איך מצטרפים אליו?",
            (
                "אצל הראל יש שלושה מוצרי ביטוח משכנתא: ביטוח חיים למשכנתא, "
                "משכנתא בטוחה ומשכנתא בטוחה (ב).\n\n"
                "להצטרפות לביטוח חיים למשכנתא מתקשרים ל-2735* או משאירים פרטים באתר. "
                "שים לב: ניתן לרכוש כיסויי ביטוח למשכנתא למועמדים עד גיל 60, "
                "ותקרת הסכום הכוללת שניתן לרכוש היא 2,000,000 ₪."
            ),
            "2,000,000",
        ),  # inline שים לב on the join paragraph (real champion shape)
        (
            "dev-43",
            'אושפזתי בבית חולים בחו"ל ליותר משלושה ימים. האם הביטוח ישלם על כרטיס טיסה ולינה עבור מלווה שיגיע אליי?',
            (
                'כן. אם אושפזת בחו"ל מעל 3 ימים, המבטח ישלם למלווה אחד עד $2,500, '
                "בהשתתפות עצמית של $50.\n\n"
                'שים לב: בפוליסת "שיר לנוסע" הכיסוי דומה אך מוגבל לכ-$10,000 במצטבר.'
            ),
            "10,000",
        ),
        (
            "dev-46",
            'אני בחו"ל וגיליתי רק עכשיו, בשבוע 14, שאני בהיריון. האם ההוצאות הרפואיות שלי בקשר להיריון מכוסות בפוליסת דרכון פרימיום?',
            (
                "לא. בפוליסה הבסיסית מכוסים הוצאות רפואיות בגין היריון עד שבוע 12 "
                "(כולל) שאובחן לראשונה בחו\"ל — ומכיוון שההיריון התגלה בשבוע 14, "
                "ההוצאות בקשר אליו לא יכוסו.\n\n"
                "לידיעתך: הרחבה להיריון (שמכסה עד שבוע 32 למבוטחת עד גיל 42) "
                "ניתנת לרכישה רק לפני שבוע 32 ובתשלום נוסף."
            ),
            "32",
        ),
    ]
    for name, q, a, banned in cases:
        ok, reason = should_verify(q, a)
        assert ok, f"{name} should gate-in: {reason}"
        new, removed = strip_unasked_asides(q, a)
        assert removed, f"{name} should strip aside"
        assert banned not in new, f"{name} still has {banned}"
        assert covers_question(q, a, new)[0], f"{name} lost question coverage"


def test_protect_garden_cap() -> None:
    q = (
        "הדירה שלי מבוטחת בסכום של 1,200,000 ש\"ח. שריפה פגעה בגינה — "
        "בדשא, בשיחים ובמערכת ההשקיה. מה הסכום המקסימלי שהביטוח ישלם על הנזק הזה?"
    )
    a = (
        "כן, נזק למדשאה מכוסה בשריפה, ואחריות המבטח מוגבלת ל־2% מסכום ביטוח הדירה. "
        "לכן הסכום המקסימלי: 2% × 1,200,000 = 24,000 ₪."
    )
    # Even if LLM flagged the calc sentence, protect Q numbers
    filtered = filter_protected_spans(
        q, a, ["לכן הסכום המקסימלי: 2% × 1,200,000 = 24,000 ₪."]
    )
    assert filtered == []
    # covers_question blocks nuking the amount that uses Q's number
    stripped, removed = strip_unsupported(a, ["2% × 1,200,000 = 24,000"])
    if removed:
        assert not covers_question(q, a, stripped)[0]


def test_revert_empty() -> None:
    ok, reason = covers_question("כמה?", "הסכום הוא 100", "")
    assert not ok and reason == "empty_after"


def main() -> None:
    test_strip_extras_41_43_46()
    test_protect_garden_cap()
    test_revert_empty()
    print("rag.test_verify: OK")


if __name__ == "__main__":
    main()
