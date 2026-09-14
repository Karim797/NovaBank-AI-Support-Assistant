---
doc_id: virtual-and-disposable-cards
title: Virtual and Disposable Cards
topic: cards
version: 2026-01-15
---

## Virtual cards

A virtual card is a card number issued instantly in the app for online payments.
Every customer can hold up to 5 standard virtual cards at no cost. They draw on the
same balance as the physical card and share the account limits in `limits`.

## Disposable virtual cards

A disposable card regenerates its number after every successful payment, which
protects you from merchants storing card details. You can hold up to 20 disposable
cards; the number regenerates immediately after each transaction, so a merchant
cannot bill the same number twice.

## When a virtual card does not work

Common causes, in order of frequency: the merchant requires a physical card
imprint; the card was frozen or deleted; a disposable card number was reused for a
recurring payment; or the transaction exceeds the limits in `limits`. Disposable
cards must never be used for subscriptions, hotel pre-authorisations, car hire or
airline bookings, because those merchants re-bill the original number and the
payment will be declined.

## Adding cards to Apple Pay and Google Pay

Physical and standard virtual cards can be added to Apple Pay and Google Pay from
Cards > Digital wallets. Disposable cards cannot be added to a wallet. Wallet
payments use the same limits and appear in the app in real time.

## Deleting a card

Deleting a virtual card is permanent and immediate. Any pending authorisation on a
deleted card still settles; any recurring payment linked to it will fail. Move
recurring payments to your physical card or a standard virtual card first.
