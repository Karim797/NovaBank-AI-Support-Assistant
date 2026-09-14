---
doc_id: card-activation-and-pin
title: Card Activation, PIN and Passcode
topic: cards
version: 2026-01-15
---

## Activating a new card

Open Cards > select the card > Activate, then enter the 6-digit code printed on
the card carrier. Activation is instant. A card that is not activated within 60
days of dispatch is cancelled automatically and must be re-ordered.

## Viewing and changing your PIN

Your PIN is generated when the card is produced and can be viewed in Cards > PIN
(re-authentication required). To change it, use any NovaBank-supported ATM: insert
the card, choose PIN services, then Change PIN. PINs cannot currently be changed in
the app. A changed PIN takes effect immediately.

## Blocked PIN

Three consecutive incorrect PIN entries block the card for ATM and chip-and-PIN
use. Unblock it in Cards > PIN > Unblock PIN; the block clears within 5 minutes.
Contactless payments continue to work while a PIN is blocked, up to the contactless
limit in `limits`. A PIN cannot be unblocked more than 3 times in 30 days; after
that the card must be replaced.

## Forgotten app passcode

If you forget your app passcode, choose Forgotten passcode on the login screen.
You will re-verify with your registered device and a one-time code sent by SMS, then
set a new passcode. If you no longer have the registered device, you must complete
full identity re-verification (see `identity-verification-kyc`), which takes up to
2 business days.

## Contactless not working

Contactless stops working when: the card has not yet been used once with a PIN,
the cumulative contactless limit since the last PIN entry has been reached, or the
card is frozen. Using the card once at a chip-and-PIN terminal resets the
cumulative counter. If contactless still fails after a successful PIN transaction,
the card antenna may be damaged and the card should be replaced.
