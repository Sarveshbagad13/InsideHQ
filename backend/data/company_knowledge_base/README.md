# NovaSphere Technologies — Synthetic Company Knowledge Base

This is a fictional enterprise knowledge base designed for RAG ingestion and evaluation.

It contains 35 Markdown documents covering:
- Company
- HR
- Engineering
- Product
- Finance
- Security
- Support

All company, employee, policy, product, and operational information is fictional.

## Suggested ingestion metadata

For each document, derive:
- department
- document_name
- document_type
- source
- access_level
- version
- effective_date

Example:
{
  "department": "HR",
  "document_type": "policy",
  "source": "leave_policy.md",
  "access_level": "employee"
}

## Suggested RAG test questions

1. How many paid leave days does a full-time employee receive?
2. How many annual leave days can be carried forward?
3. What are the core collaboration hours for remote employees?
4. What is the default API rate limit?
5. What happens during a P1 incident?
6. Who handles billing and invoice issues?
7. What purchases require Finance review?
8. What is the learning allowance?
9. What products does NovaSphere offer?
10. What was introduced in FlowOps 4.8?
11. What is the response target for a P2 support ticket?
12. What happens when an employee changes roles?
13. What database does NovaSphere use for transactional data?
14. Who approves employee reimbursement claims?
15. What should an engineer include when escalating a support issue?
16. Can enterprise pricing be determined from the standard pricing document?
17. What should an employee do if they suspect a security incident?
18. What are the current product roadmap priorities?
19. What is the difference between Business and Enterprise plans?
20. Can roadmap priorities change?

## Important evaluation cases

Your assistant should distinguish:
- current policy vs planning information
- standard pricing vs negotiated enterprise pricing
- support SLA vs enterprise contractual SLA
- technical troubleshooting vs security incidents
- employee policy vs product documentation

The dataset intentionally has related information spread across multiple documents so that retrieval and multi-document reasoning can be tested.
