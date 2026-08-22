"""Second expansion, aimed at what the error analysis actually found.

Cross-validated accuracy after the first expansion was 82.0% +/- 9.5% for
category. The +/- 9.5% is a sample size problem, not a model problem - at 228
rows any single split lands ten points either side of the truth. Nothing but
more data narrows that band.

WHAT THE MISTAKES POINTED AT
Reading every misclassified ticket in the held-out set gave three causes:

  rare vocabulary   "cors", "depot", "console" appeared once or twice in 182
                    training rows, so they carried no weight. Fixed by writing
                    more tickets that use the working vocabulary of each area.

  lexical collision "do you deliver on bank holidays" was classified billing,
                    because "bank" appears in "bank transfer". The fix is more
                    examples of both senses so the surrounding words decide.

  genuine ambiguity "can i get a copy of the receipt sent to a second email" is
                    billing or account depending on how you read it. Roughly one
                    ticket in twenty is like this, which puts a ceiling near 95%
                    that no amount of data removes.

SENTIMENT BALANCE
positive had 29 examples against 139 negative. A classifier facing that split
learns that guessing negative is usually safe. This batch is deliberately
weighted towards neutral and positive to bring the majority class down from 61%.
"""

import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "tickets.csv"

# (text, category, priority, sentiment)
TICKETS: list[tuple[str, str, str, str]] = [
    # ---------------------------------------------------------------- billing
    ("Statement shows a charge I do not recognise from the 3rd", "billing", "high", "negative"),
    ("Annual renewal took payment despite my cancellation request", "billing", "high", "negative"),
    ("Refund was approved but nothing has reached my bank", "billing", "high", "negative"),
    ("Paid by bank transfer last week and the account still shows unpaid", "billing", "high", "negative"),
    ("Charged for five seats when we only have three people", "billing", "high", "negative"),
    ("Overdue notice arrived for an invoice I settled in June", "billing", "high", "negative"),
    ("Card expired and now the service is suspended, need to fix urgently", "billing", "high", "negative"),
    ("Two invoices issued for the same billing period", "billing", "high", "negative"),
    ("Currency conversion added a fee nobody mentioned", "billing", "medium", "negative"),
    ("Upgrade charged the full amount rather than the difference", "billing", "medium", "negative"),
    ("Promotional rate ended without notice", "billing", "medium", "negative"),
    ("Purchase order number is missing from the invoice", "billing", "medium", "neutral"),
    ("Please send invoices to our finance mailbox instead", "billing", "medium", "neutral"),
    ("How do I switch the payment method to a company card", "billing", "medium", "neutral"),
    ("Can we move the renewal date to the start of our financial year", "billing", "medium", "neutral"),
    ("Need the last four invoices as a single PDF", "billing", "medium", "neutral"),
    ("What is your policy on refunds after fourteen days", "billing", "low", "neutral"),
    ("Do you charge extra for additional storage", "billing", "low", "neutral"),
    ("Is there a setup fee for new accounts", "billing", "low", "neutral"),
    ("How is usage measured for the metered plan", "billing", "low", "neutral"),
    ("Can I see a breakdown of what each line item covers", "billing", "low", "neutral"),
    ("Would a two year commitment reduce the price", "billing", "low", "neutral"),
    ("Do you accept purchase orders from public sector bodies", "billing", "low", "neutral"),
    ("What happens to billing if we pause the account for a month", "billing", "low", "neutral"),
    ("Is VAT charged for customers outside the UK", "billing", "low", "neutral"),
    ("Can the invoice show our registered company number", "billing", "low", "neutral"),
    ("Billing team sorted my duplicate charge within an hour, excellent", "billing", "low", "positive"),
    ("Really clear invoices, easy to reconcile, thank you", "billing", "low", "positive"),
    ("Happy with the plan and considering adding more seats", "billing", "low", "positive"),
    ("Thanks for honouring the old rate, that was generous", "billing", "low", "positive"),
    ("The new billing dashboard is a big improvement", "billing", "low", "positive"),
    ("Appreciate the reminder email before renewal, very helpful", "billing", "low", "positive"),
    ("Pleased with how simple the upgrade process was", "billing", "low", "positive"),
    ("Grateful for the prorated credit, that resolved it", "billing", "low", "positive"),
    ("Direct debit mandate has been cancelled by my bank", "billing", "medium", "negative"),
    ("Discount was applied to the wrong line on the invoice", "billing", "medium", "negative"),
    ("Can I get an itemised receipt for expenses", "billing", "low", "neutral"),
    ("Does the plan include support or is that separate", "billing", "low", "neutral"),
    ("Invoice arrived in the wrong currency again", "billing", "medium", "negative"),
    ("Payment page rejects my card without giving a reason", "billing", "high", "negative"),
    ("How far back can I download billing history", "billing", "low", "neutral"),

    # --------------------------------------------------------------- shipping
    ("Courier attempted delivery without ringing the bell", "shipping", "medium", "negative"),
    ("Consignment has been sitting at the depot for four days", "shipping", "high", "negative"),
    ("Driver marked it as refused but nobody was asked", "shipping", "high", "negative"),
    ("Pallet arrived with three cartons water damaged", "shipping", "high", "negative"),
    ("Shipment split across two deliveries without warning", "shipping", "medium", "negative"),
    ("Goods delivered to reception and nobody signed for them", "shipping", "medium", "negative"),
    ("Order dispatched to the billing address not the delivery address", "shipping", "high", "negative"),
    ("Overnight service arrived three days late", "shipping", "high", "negative"),
    ("Carrier lost the parcel and is not responding", "shipping", "high", "negative"),
    ("Return was collected two weeks ago and not yet credited", "shipping", "high", "negative"),
    ("Can I collect the order from the depot myself", "shipping", "medium", "neutral"),
    ("Is a delivery on a bank holiday weekend possible", "shipping", "low", "neutral"),
    ("What are your cut off times for next day dispatch", "shipping", "low", "neutral"),
    ("Do you offer a timed delivery slot for large items", "shipping", "low", "neutral"),
    ("Which carrier handles deliveries to the Highlands", "shipping", "low", "neutral"),
    ("Can the courier call ahead before arriving", "shipping", "low", "neutral"),
    ("How do I add loading bay instructions to the order", "shipping", "medium", "neutral"),
    ("Is a signature required on delivery", "shipping", "low", "neutral"),
    ("Can I redirect a parcel that has already been dispatched", "shipping", "medium", "neutral"),
    ("Do you deliver to BFPO addresses", "shipping", "low", "neutral"),
    ("What size vehicle should we expect for a pallet delivery", "shipping", "low", "neutral"),
    ("How do I book a collection for a warranty return", "shipping", "medium", "neutral"),
    ("Are weekend deliveries available in central London", "shipping", "low", "neutral"),
    ("Can two orders be consolidated into a single consignment", "shipping", "low", "neutral"),
    ("Do you provide proof of delivery on request", "shipping", "low", "neutral"),
    ("Delivery arrived exactly in the slot promised, very good", "shipping", "low", "positive"),
    ("Driver was helpful and carried it inside, much appreciated", "shipping", "low", "positive"),
    ("Packaging was excellent, nothing damaged at all", "shipping", "low", "positive"),
    ("Fast dispatch and clear tracking, no complaints", "shipping", "low", "positive"),
    ("Pleased you now offer Saturday delivery", "shipping", "low", "positive"),
    ("Returns process was painless, thanks", "shipping", "low", "positive"),
    ("Great to see plastic free packaging on the last order", "shipping", "low", "positive"),
    ("Courier rescheduled easily when I was out, very flexible", "shipping", "low", "positive"),
    ("Tracking page has not updated since it left the warehouse", "shipping", "medium", "negative"),
    ("Wrong quantity delivered, ordered ten received four", "shipping", "high", "negative"),
    ("Parcel left in a bin store and now missing", "shipping", "high", "negative"),
    ("Delivery note does not match what was in the box", "shipping", "medium", "negative"),
    ("Can I nominate a safe place for future deliveries", "shipping", "low", "neutral"),
    ("How long do you hold an undelivered parcel at the depot", "shipping", "low", "neutral"),
    ("Is there a surcharge for offshore postcodes", "shipping", "low", "neutral"),

    # ---------------------------------------------------------------- account
    ("Cannot sign in, it says my credentials are invalid", "account", "high", "negative"),
    ("Account locked after a password manager filled the wrong entry", "account", "high", "negative"),
    ("Colleague still has access three months after leaving", "account", "high", "negative"),
    ("Login page rejects the code from my authenticator app", "account", "high", "negative"),
    ("Received a password change email I did not request", "account", "high", "negative"),
    ("Workspace disappeared from my list this morning", "account", "high", "negative"),
    ("Invitation link says it has already been used", "account", "medium", "negative"),
    ("Cannot remove a user who no longer works here", "account", "medium", "negative"),
    ("Email change request never sent the confirmation", "account", "medium", "negative"),
    ("Signed out unexpectedly several times today", "account", "medium", "negative"),
    ("How do I set up single sign on with our identity provider", "account", "medium", "neutral"),
    ("Can I restrict access by IP address", "account", "medium", "neutral"),
    ("What is the process for transferring account ownership", "account", "medium", "neutral"),
    ("How do I create a read only role for auditors", "account", "medium", "neutral"),
    ("Is there a way to enforce two factor for everyone", "account", "medium", "neutral"),
    ("Can I see when each member last signed in", "account", "low", "neutral"),
    ("How many seats does the current plan include", "account", "low", "neutral"),
    ("Where do I find the account identifier for support", "account", "low", "neutral"),
    ("Can I rename the workspace after creating it", "account", "low", "neutral"),
    ("Is there an option to hide my email from other members", "account", "low", "neutral"),
    ("How do I set a default timezone for the team", "account", "low", "neutral"),
    ("Can I use the same login across two organisations", "account", "low", "neutral"),
    ("What is your data retention policy for closed accounts", "account", "low", "neutral"),
    ("Do you support SCIM provisioning", "account", "low", "neutral"),
    ("How do I download an export of everything before leaving", "account", "medium", "neutral"),
    ("Onboarding was very smooth, the whole team is set up", "account", "low", "positive"),
    ("Support helped me recover access quickly, thank you", "account", "low", "positive"),
    ("Permissions model is well thought out, easy to configure", "account", "low", "positive"),
    ("Two factor setup took two minutes, nicely done", "account", "low", "positive"),
    ("Really like being able to see the login history", "account", "low", "positive"),
    ("Glad you added single sign on, that was the blocker for us", "account", "low", "positive"),
    ("Invitation flow is clear and my team had no questions", "account", "low", "positive"),
    ("Thanks for extending the trial while we finished procurement", "account", "low", "positive"),
    ("Recovery email is an old address I can no longer access", "account", "high", "negative"),
    ("New member cannot accept the invitation, link expires instantly", "account", "high", "negative"),
    ("Profile changes revert after a few minutes", "account", "medium", "negative"),
    ("Is there a limit on the number of workspaces", "account", "low", "neutral"),
    ("Can I merge a personal account into the company one", "account", "medium", "neutral"),
    ("Notification settings do not persist between sessions", "account", "medium", "negative"),
    ("Do you offer audit logs for compliance reporting", "account", "low", "neutral"),

    # -------------------------------------------------------------- technical
    ("Browser console shows a CORS error when loading the widget", "technical", "high", "negative"),
    ("API returns 500 on every request to the reports endpoint", "technical", "high", "negative"),
    ("Webhook deliveries stopped without any change on our side", "technical", "high", "negative"),
    ("Bulk import times out after processing about two hundred rows", "technical", "high", "negative"),
    ("Search index appears stale, new records do not appear", "technical", "high", "negative"),
    ("File upload returns a 413 for anything over two megabytes", "technical", "high", "negative"),
    ("OAuth callback fails with an invalid redirect uri", "technical", "high", "negative"),
    ("Rate limit headers are missing from API responses", "technical", "medium", "negative"),
    ("Pagination returns duplicate records across pages", "technical", "high", "negative"),
    ("Timestamps come back in UTC but the docs say local", "technical", "medium", "negative"),
    ("Dashboard charts do not render in Firefox", "technical", "medium", "negative"),
    ("Cache seems stale, changes take an hour to appear", "technical", "medium", "negative"),
    ("Editor loses formatting when pasting from a document", "technical", "medium", "negative"),
    ("Mobile app crashes when opening the attachments tab", "technical", "high", "negative"),
    ("Scheduled export produced an empty file this week", "technical", "high", "negative"),
    ("Sorting by date puts blank values first", "technical", "low", "negative"),
    ("Is there a staging environment for testing integrations", "technical", "low", "neutral"),
    ("What is the rate limit on the public API", "technical", "low", "neutral"),
    ("Do you publish an OpenAPI specification", "technical", "low", "neutral"),
    ("Which webhook events are available", "technical", "low", "neutral"),
    ("How do I rotate an API key without downtime", "technical", "medium", "neutral"),
    ("Is there a bulk endpoint for creating records", "technical", "low", "neutral"),
    ("What is the maximum payload size for the API", "technical", "low", "neutral"),
    ("Do you support webhooks with a signing secret", "technical", "low", "neutral"),
    ("How long are logs retained in the console", "technical", "low", "neutral"),
    ("Is there a browser extension available", "technical", "low", "neutral"),
    ("Can the export include custom fields", "technical", "low", "neutral"),
    ("Does the API support filtering by date range", "technical", "low", "neutral"),
    ("The latest release fixed the import bug, thanks for the quick work", "technical", "low", "positive"),
    ("API is well designed and the docs are genuinely good", "technical", "low", "positive"),
    ("Performance improvement this week is very noticeable", "technical", "low", "positive"),
    ("New filtering options are exactly what we needed", "technical", "low", "positive"),
    ("Integration took an afternoon, very straightforward", "technical", "low", "positive"),
    ("Appreciate the detailed changelog on every release", "technical", "low", "positive"),
    ("Webhook retries saved us during an outage, well designed", "technical", "low", "positive"),
    ("Really pleased with the new bulk edit feature", "technical", "low", "positive"),
    ("Session token expires far sooner than documented", "technical", "medium", "negative"),
    ("Error messages do not say which field was rejected", "technical", "medium", "negative"),
    ("Console throws a null reference when the list is empty", "technical", "medium", "negative"),
    ("Is there a sandbox key for automated tests", "technical", "low", "neutral"),
]


def main() -> None:
    existing = list(csv.DictReader(DATA.open()))
    seen = {row["text"].strip().lower() for row in existing}
    next_id = max(int(row["ticket_id"]) for row in existing) + 1

    added, duplicates = [], 0
    for text, category, priority, sentiment in TICKETS:
        if text.strip().lower() in seen:
            duplicates += 1
            continue
        seen.add(text.strip().lower())
        added.append({"ticket_id": next_id, "text": text, "category": category,
                      "priority": priority, "sentiment": sentiment})
        next_id += 1

    with DATA.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=["ticket_id", "text", "category",
                                      "priority", "sentiment"]).writerows(added)

    print(f"added {len(added)} ticket(s), skipped {duplicates} duplicate(s)")
    print(f"dataset now {len(existing) + len(added)} rows")


if __name__ == "__main__":
    main()
