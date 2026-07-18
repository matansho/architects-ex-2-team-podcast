# Stage 1 Report

Stage 1 covered a bare open-weights model on the dev set (no documents), a custom LLM-as-judge evaluation harness (relevance, hallucination, refusal, citation), and several prompt strategies. Answers to the three reflection questions below.

## 1. Where does the baseline succeed without any Harel documents? What does that tell you about its training data?

The model does well on questions answerable from general knowledge — Israeli insurance law and regulation, universal insurance principles and logic, and simple reasoning or arithmetic — and it is fluent in the domain's Hebrew. It fails on facts specific to Harel's own policies (exact amounts, limits, waiting periods, contact details, product names), which live only in the documents. This tells us its training data holds broad, public insurance and legal knowledge but none of Harel's proprietary policy details: it knows how insurance *works*, not what a particular Harel document *says*. The implication is that retrieval, not better prompting, is what can lift accuracy on the document-specific questions.

## 2. When it's wrong, is it wrong confidently? Which failure is worse for an insurer — a wrong answer or "I don't know"?

Yes — when it doesn't know, it rarely admits it. It invents specific, checkable facts (numbers, sums, percentages, contact details) and states them just as confidently as its correct answers, so a reader can't tell them apart. For an insurer a confident wrong answer is worse than "I don't know": a fabricated figure or address misleads the customer and creates compliance and liability exposure, whereas an honest "check your policy" is safe and points them to an authoritative source.

## 3. The judge is itself an LLM. Find one question where you disagree with the judge's verdict. What does that imply about your evaluation at scale?

In one question (dev-26-health-easy) the model added invented specifics — a made-up numeric limit and a fabricated list — that the judge accepted as *not* a hallucination because they did not directly contradict the reference, even though inventing an unverifiable specific is exactly the failure we care about. So the LLM judge can miss fabricated facts and is inconsistent at the margins (since it's an LLM). The takeaway is that the metric is a useful compass, not an exact ruler: it can under-count hallucination and shifts between runs, so we trust large, repeated differences and treat small ones with caution.
