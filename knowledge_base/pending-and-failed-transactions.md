---
doc_id: pending-and-failed-transactions
title: Pending and Failed Transactions
topic: transactions
version: 2026-01-15
---

## What "pending" means

A pending transaction is an authorisation: the merchant has reserved the money but
has not claimed it. The amount is deducted from your available balance but has not
left the account. Pending entries can change amount when they settle — for example
a fuel pump reserves GBP 100 and settles at the amount actually pumped.

## How long pending transactions take to clear

- Card purchases: usually 1 to 3 business days, up to 7 days maximum.
- ATM withdrawals: usually up to 3 business days, always within 7 days.
- Hotels, car hire and cruise pre-authorisations: up to 14 days.
- Top-ups by card: minutes; see `top-ups-and-deposits`.
- Transfers: see `transfers-domestic-and-international`.

If a card authorisation is not claimed within its window it expires and the money
returns to your available balance automatically. NovaBank cannot release a pending
authorisation early — only the merchant can.

## Pending transfer

An outgoing transfer stays pending while it is screened or scheduled. Screening
takes up to 30 minutes and applies to new payees, large amounts, and payments to
higher-risk destinations. A transfer scheduled for a future date shows as pending
until the send date.

## Failed transactions

A failed transaction is one that was rejected before any money moved: no
authorisation exists and no money is reserved. Failed card payments are covered by
`card-payments-and-declines`; failed transfers by
`transfers-domestic-and-international`; failed top-ups by `top-ups-and-deposits`.

## Balance not updated after a payment in

Incoming money appears as soon as NovaBank receives it. If a sender says money was
sent but it has not arrived: for internal NovaBank transfers it is instant, so
check the recipient details; for Faster Payments allow up to 2 hours; for SEPA and
SWIFT see the timings in `transfers-domestic-and-international`. Money sent to a
closed or incorrect account is returned to the sender by their bank, typically
within 5 business days.

## Balance not updated after a cheque or cash deposit

NovaBank does not accept cheques or cash deposits at all. There is no cheque
imaging service and no branch or Post Office cash-in facility. Money can only enter
your account by bank transfer, card top-up, or an incoming payment; see
`top-ups-and-deposits`. If you sent a cheque to NovaBank it will not be processed.
