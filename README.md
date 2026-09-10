# Hiver AI Support Agent

An AI-assisted customer support prototype built for the Hiver SDE Intern take-home assignment.

The system uses historical AmazonHelp customer-support conversations to classify customer intent, retrieve similar resolved cases, generate a historically grounded response, and decide whether the request should be auto-handled or escalated to a human.

---

## 1. Problem Statement

Customer-support data from Twitter is noisy, conversational, and difficult to use directly.

The goal of this project is to build a support-assistance pipeline that can:

1. Understand the customer's request.
2. Classify it into a small set of support intents.
3. Retrieve similar historical customer-support conversations.
4. Suggest a response grounded in historical support behavior.
5. Decide whether the request is safe to auto-handle or should be escalated.
6. Provide evidence explaining the decision.

The selected brand for this project is **AmazonHelp**.

---

## 2. What Good Looks Like

A useful support system should:

- Correctly identify the customer's intent.
- Retrieve relevant historical cases.
- Produce responses consistent with previous support behavior.
- Avoid inventing unsupported information.
- Escalate sensitive or uncertain cases.
- Provide evidence for its recommendation.
- Be measurable using a held-out evaluation set.

The system prioritizes **safe assistance over blindly automating every request**.

---

## 3. Dataset

The project uses the Kaggle:

**Customer Support on Twitter** dataset.

The original dataset contains approximately 2.8 million tweets.

After filtering for AmazonHelp conversations and linking customer messages with AmazonHelp responses, the project produced:

- **168,814 AmazonHelp customer-support conversations**
- Customer message
- Historical AmazonHelp reply

The dataset contains both inbound customer messages and support responses.

For this project:

- `inbound = True` → customer message
- `inbound = False` → support response

---

## 4. System Architecture

```text
Customer Message
       |
       v
Intent Classification
       |
       v
Retrieve Similar Historical Cases
       |
       v
Historical Evidence
       |
       v
Generate Grounded Reply
       |
       v
Auto-Handle / Escalate
       |
       v
Final Support Recommendation