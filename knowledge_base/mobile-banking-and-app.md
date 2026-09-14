---
doc_id: mobile-banking-and-app
title: The NovaBank App
topic: app
version: 2026-01-15
---

## Getting the app and signing in

NovaBank is app-only: iOS 16 or later, Android 10 or later. There is no web
banking and no telephone banking beyond the fraud line. Sign in with biometrics or
a 6-digit passcode; see `card-activation-and-pin` for a forgotten passcode.

## One device at a time

An account is active on one device at a time. Signing in on a new device
de-registers the old one and requires a one-time code to your registered mobile
number, plus a selfie check if the device is new to NovaBank. If you no longer have
your registered number, you must complete identity re-verification (see
`identity-verification-kyc`).

## Statements and transaction history

Transaction history is unlimited in the app. Monthly PDF statements are generated
on the 1st of each month and are available for 12 months after account closure.
Statements are digital only.

## Notifications

Payment notifications are instant. If notifications stop, check device settings,
then Profile > Notifications. Notification failure never affects the payment
itself.

## Linking cards from other banks

Card linking lets you see balances from other UK banks in the NovaBank app through
Open Banking. Linked accounts are read-only: you cannot spend from them, and the
link expires every 90 days under Open Banking rules and must be renewed.

## App not working

Try: force-close and reopen, check for an app update, confirm the device meets the
minimum OS version, and check the status page at status.novabank.example. If a
payment appears to have failed in the app, always check the transaction list before
retrying — the payment may have gone through.
