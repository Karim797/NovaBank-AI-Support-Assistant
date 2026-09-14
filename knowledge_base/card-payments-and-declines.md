---
doc_id: card-payments-and-declines
title: Card Payments, Declines and Acceptance
topic: payments
version: 2026-01-15
---

## Where NovaBank cards are accepted

NovaBank issues Visa debit cards, accepted anywhere Visa is accepted, in any
currency. Merchants that only accept Mastercard or American Express cannot take a
NovaBank card. Some merchants (typically car hire, hotels and fuel pumps) place a
pre-authorisation hold; see `pending-and-failed-transactions`.

## Why a card payment was declined

The most common reasons a payment is declined are: insufficient balance in the
currency being charged; the card is frozen; a daily or per-transaction limit in
`limits` was exceeded; the card has expired or was never activated; the PIN is
blocked; the merchant category is blocked in Cards > Payment controls (for example
gambling); or the transaction failed the fraud check.

## Checking why your specific payment failed

Every declined attempt appears in the app timeline with a decline reason. Open the
transaction and read the reason line before retrying — repeating the same payment
without changing anything will usually decline again, and more than 5 declines in
an hour temporarily locks card payments for 30 minutes.

## Payments you do not recognise

If a card payment appears that you did not make, follow
`duplicate-and-unrecognised-charges`. Note that merchant trading names often differ
from the shopfront name, and subscription renewals may be billed by a payment
processor rather than the merchant, so check the merchant details in the
transaction before disputing.

## Direct debits

A direct debit lets a merchant collect from your GBP balance. Manage mandates in
Payments > Direct debits. You can cancel a mandate at any time, but cancelling
does not cancel the underlying contract with the merchant. A direct debit taken
after you cancelled the mandate, or for the wrong amount, is covered by the Direct
Debit Guarantee and is refunded immediately on request.

## Reverted or reversed card payments

A reverted payment is one the merchant cancelled before settlement. The
authorisation hold is released and the money returns to your available balance;
no refund is issued because no money left the account. Reversals appear as the
original pending entry disappearing, usually within 3 business days.
