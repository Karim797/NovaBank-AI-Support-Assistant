---
doc_id: top-ups-and-deposits
title: Topping Up and Depositing Money
topic: topups
version: 2026-01-15
---

## Ways to add money

1. **Bank transfer** from an account in your own name — free, arrives within 2
   hours by Faster Payments.
2. **Debit card top-up** in the app — instant, free up to the free card top-up
   allowance of GBP 2,000 per rolling 30 days, then the card top-up fee of 1%.
3. **Incoming payments** from third parties — free.

NovaBank does not accept cash or cheque deposits by any route, and there is no
Post Office or PayPoint cash-in partner.

## Automatic top-up

Automatic top-up refills your balance from a linked debit card when it falls below
a threshold you set (Profile > Automatic top-up). Choose a trigger balance and a
top-up amount. Automatic top-ups count towards the free card top-up allowance and
the card top-up limit in `limits`. Automatic top-up pauses after two consecutive
failures and must be re-enabled manually.

## Top-up failed

Causes, in order of frequency: the source card was declined by its issuer; the card
top-up limit in `limits` was reached; the source card is a credit card (not
supported); the card is not in your own name (not supported); or 3-D Secure
verification was not completed. A failed top-up never debits your source card, but
the source bank may show a pending authorisation that clears within 7 days.

## Top-up reverted

A top-up is reverted when the source bank recalls it after crediting, usually
because the source card or account was reported compromised, or the top-up was
flagged by the source bank's fraud checks. The amount is deducted from your balance
and you are notified with the reason. If this pushes your balance negative, the
account is frozen for incoming payments only until the shortfall is cleared.

## Verifying a top-up

Top-ups above GBP 5,000 in a rolling 30-day period, or a single top-up above GBP
2,500 from a newly linked card, trigger a verification check. You may be asked to
confirm the source card belongs to you and, above the thresholds in
`identity-verification-kyc`, to provide source of funds evidence. Funds are held,
not lost, while verification completes — normally within 1 business day.

## Balance not updated after a bank transfer

Faster Payments normally arrive within 2 hours. If more than 2 hours have passed:
confirm the sender used the correct sort code and account number from Home >
Account details; check the sending bank has actually released the payment (many
show it as sent immediately while batching it); and note that transfers from
building societies and some business accounts can take a full business day. If the
details were wrong, the sending bank must recall the payment — NovaBank cannot
retrieve money credited to another customer's account.
