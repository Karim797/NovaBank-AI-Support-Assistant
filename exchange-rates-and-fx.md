---
doc_id: exchange-rates-and-fx
title: Exchange Rates and Currency Conversion
topic: fx
version: 2026-01-15
---

## The rate NovaBank uses

NovaBank converts at the interbank mid-market rate available at the moment the
conversion is processed. The rate shown in the app before you confirm an in-app
exchange is held for 60 seconds. For card payments and ATM withdrawals in a
currency you do not hold, the rate applied is the rate at the time the transaction
is authorised, not when it settles — so a settled amount can differ slightly from
the pending amount.

## Conversion costs

Conversions are free up to the free FX allowance of GBP 1,000 per calendar month.
Above the allowance the FX conversion fee of 0.5% applies. Conversions made on
Saturday or Sunday also carry the weekend FX markup of 0.5%, because interbank
markets are closed and NovaBank carries the weekend rate risk. Both are listed in
`fees-and-charges` and shown as separate line items.

## Exchanging in the app

Exchange money between your currency balances in Home > Exchange. Exchanges are
instant and irreversible once confirmed. Exchanging into a currency you do not yet
hold creates that balance automatically, up to the maximum of 3 currency balances
in `limits`.

## Supported currencies

You can hold and exchange GBP, EUR and USD. NovaBank can convert into a further 25
currencies at the point of a card payment or ATM withdrawal, but does not hold them
as balances. NovaBank does not support cryptocurrency, does not offer crypto
exchange, and does not support currencies subject to sanctions or capital controls.

## "Wrong" exchange rate on a card payment

Three things commonly explain a rate that looks wrong:

1. **Dynamic currency conversion.** If the merchant or ATM offered to charge you in
   GBP and you accepted, they converted at their own rate, not NovaBank's. Always
   choose the local currency.
2. **Authorisation versus settlement.** The rate is fixed at authorisation; if the
   merchant settles days later the market has moved, but NovaBank does not re-rate.
3. **Weekend markup.** A conversion on Saturday or Sunday includes the weekend
   markup.

Open the transaction to see the rate applied, the base amount and each fee line. If
the rate shown does not match the interbank rate at the authorisation timestamp,
report it and NovaBank will re-rate the transaction and refund the difference.
