---
doc_id: transfers-domestic-and-international
title: Transfers, Domestic and International
topic: transfers
version: 2026-01-15
---

## Transfer types and timings

- NovaBank to NovaBank: instant, 24/7, free.
- UK Faster Payments (GBP): usually under 2 hours, always within 1 business day, free.
- SEPA (EUR, supported countries): 1 business day, free.
- SWIFT international: 1 to 5 business days, charged at the international transfer
  fee in `fees-and-charges` plus any correspondent bank deductions.

Cut-off for same-day SWIFT is 14:00 UK time on a business day. Transfers submitted
after the cut-off, at weekends or on UK bank holidays are processed the next
business day.

## Adding a beneficiary

Add payees in Payments > New payee. The payee name is checked against the receiving
bank's records (Confirmation of Payee) for UK accounts. A partial or failed name
match does not block the payment, but you must confirm you want to continue.

## Beneficiary not allowed

Some beneficiaries are rejected outright: accounts in unsupported or sanctioned
countries; currencies NovaBank does not support (see `exchange-rates-and-fx`);
payments to your own account at an unverified third-party institution; and payments
to merchant categories blocked on your account. The rejection reason is shown when
you add the payee. Sanctions-related rejections cannot be appealed in the app and
require contacting support.

## Transfer declined or failed

Common causes: insufficient balance including the fee; the daily transfer limit in
`limits` was exceeded; incorrect IBAN, sort code or account number; the payee bank
rejected the payment; or the payment failed compliance screening. Failed transfers
are returned to your balance the same business day. SWIFT payments returned by the
receiving bank can take up to 10 business days and the original transfer fee is
not refunded.

## Cancelling a transfer

A transfer can be cancelled while it is pending — open it and choose Cancel. Once
it has left NovaBank it cannot be cancelled; NovaBank can send a recall request to
the receiving bank, which has no obligation to return the money. Recall requests
take up to 20 business days and only succeed if the funds are still available.

## Transfer not received by the recipient

Check first that the transfer is not still pending. If it has been sent, download
the payment confirmation (which includes the reference the receiving bank needs)
and give it to the recipient to trace at their end. For SWIFT payments NovaBank can
raise a SWIFT trace after 5 business days; traces take up to 10 further business
days. Intermediary banks may deduct their own charges from a SWIFT payment, so the
recipient may receive less than you sent.

## Receiving money

Share your account details from Home > Account details: sort code and account
number for GBP, IBAN and BIC for EUR. Incoming payments are credited as soon as
they are received. NovaBank does not charge to receive money, but the sender's bank
or an intermediary bank may deduct a fee.
