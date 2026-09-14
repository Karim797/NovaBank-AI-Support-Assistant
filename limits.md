---
doc_id: limits
title: Account and Payment Limits
topic: limits
version: 2026-01-15
---

## Limit schedule (authoritative)

This section is the single source of truth for NovaBank limits.

| Limit | Value |
|---|---|
| Card payment, per transaction | GBP 2,000 |
| Card payments, per day | GBP 5,000 |
| Contactless, per transaction | GBP 100 |
| Contactless, cumulative before PIN required | GBP 300 |
| ATM withdrawal, per day | GBP 500 |
| ATM withdrawal, per 30 days | GBP 3,000 |
| Outgoing transfer, per day | GBP 25,000 |
| Outgoing transfer, per single payment | GBP 20,000 |
| Card top-up, per 30 days | GBP 5,000 |
| Balance, maximum | GBP 250,000 |
| Standard virtual cards, active | 5 |
| Disposable virtual cards, active | 20 |
| Currency balances per customer | 3 |
| Youth account monthly spend | GBP 500 |

## Requesting a higher limit

Card and transfer limits can be raised temporarily (up to 7 days) from Profile >
Limits, subject to identity verification and, above GBP 10,000, source of funds
evidence (see `identity-verification-kyc`). Requests are reviewed within 1 business
day. ATM limits cannot be raised.

## Lowering your own limits

You can lower any limit instantly in Profile > Limits. A lowered limit takes effect
immediately; raising it again takes 24 hours as an anti-fraud measure.

## What happens when a limit is hit

The transaction is declined with the reason "limit exceeded" and no money is
reserved. Limits reset on a rolling basis from the time of the earliest counted
transaction, not at midnight, so a daily limit may free up during the day.
