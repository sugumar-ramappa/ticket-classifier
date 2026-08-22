"""Expand the ticket dataset.

The original 60 tickets gave 12-17 examples per category, which is far below
what TF-IDF plus a Random Forest needs. The priority classifier scored 25% on
three classes - worse than guessing - because it had learned nothing and was
predicting the majority label for everything.

WHY THESE ARE HAND-WRITTEN RATHER THAN TEMPLATED
Generating "my {noun} is {adjective}" from slot lists produces a dataset a model
can solve by learning the template. Accuracy goes up and means nothing, because
real tickets do not come from a template.

So these vary in length, register and vocabulary, and deliberately share words
across categories - "account" appears in billing tickets, "payment" appears in
technical ones. That overlap is what makes the task realistic, and it is exactly
what the original 60 rows lacked.

PRIORITY IS ABOUT IMPACT, NOT CATEGORY
Assigned per ticket, from what the customer cannot do:
  high    money lost, locked out, service unusable
  medium  broken but there is a way round it
  low     a question, a preference, or cosmetic

SENTIMENT NEEDED ITS OWN ATTENTION
The original 60 rows had exactly ONE positive example. A class with one example
cannot be learned - after an 80/20 split it lands wholly in train or wholly in
test, and either way the model never gets to be right about it. Positive tickets
are added here so the class exists.

Sentiment is deliberately NOT derived from priority. Making every high priority
ticket negative would produce a column the model can predict perfectly from
another column, which looks like accuracy and is really leakage. So there are
calm high priority tickets and irritated low priority ones, because that is how
people actually write.
"""

import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "tickets.csv"

# (text, category, priority, sentiment)
TICKETS: list[tuple[str, str, str, str]] = [
    # ---------------------------------------------------------------- billing
    ("I was billed twice for the same order this month", "billing", "high", "negative"),
    ("Two charges of 49.99 appeared on my statement yesterday", "billing", "high", "negative"),
    ("Refund promised three weeks ago has still not arrived", "billing", "high", "negative"),
    ("You took money after I cancelled my subscription", "billing", "high", "negative"),
    ("Card charged but the order never appeared in my history", "billing", "high", "negative"),
    ("Invoice total does not match what the checkout page showed", "billing", "high", "negative"),
    ("Charged the annual rate when I selected monthly", "billing", "high", "negative"),
    ("My card was declined but the amount still left my bank", "billing", "high", "negative"),
    ("Duplicate transaction reference 88123 needs reversing", "billing", "high", "negative"),
    ("Been overcharged VAT on every invoice since June", "billing", "high", "negative"),
    ("Subscription renewed automatically without any warning email", "billing", "medium", "negative"),
    ("Can I switch from monthly to yearly billing partway through", "billing", "medium", "neutral"),
    ("The discount code was accepted but not applied to the total", "billing", "medium", "negative"),
    ("Need a VAT invoice for accounting, the receipt is not enough", "billing", "medium", "neutral"),
    ("Please update the billing address on my next invoice", "billing", "medium", "neutral"),
    ("Why did the price go up compared to last month", "billing", "medium", "negative"),
    ("Can I pay by bank transfer instead of card", "billing", "medium", "neutral"),
    ("Receipt shows the wrong company name on it", "billing", "medium", "negative"),
    ("How do I add a second payment method to the account", "billing", "medium", "neutral"),
    ("Proration after upgrading mid cycle looks wrong", "billing", "medium", "negative"),
    ("Where can I download invoices from previous years", "billing", "low", "neutral"),
    ("Do you offer a discount for non profit organisations", "billing", "low", "neutral"),
    ("Is the price shown including tax or excluding", "billing", "low", "neutral"),
    ("What currencies do you accept for payment", "billing", "low", "neutral"),
    ("Can I get a copy of the receipt sent to a second email", "billing", "low", "neutral"),
    ("Curious whether there is a student plan available", "billing", "low", "neutral"),
    ("Does the yearly plan work out cheaper overall", "billing", "low", "neutral"),
    ("Would like to understand how usage based pricing is calculated", "billing", "low", "neutral"),
    ("Payment failed again even though my card has funds", "billing", "high", "negative"),
    ("Direct debit was taken twice this quarter", "billing", "high", "negative"),
    ("Cancelled last year and still being charged every month", "billing", "high", "negative"),
    ("Credit note issued but the balance has not changed", "billing", "medium", "negative"),
    ("Billing portal shows an amount due that I already settled", "billing", "medium", "negative"),
    ("Need the invoice split across two cost centres", "billing", "low", "neutral"),
    ("Does an unused month roll over to the next period", "billing", "low", "neutral"),

    # --------------------------------------------------------------- shipping
    ("Parcel says delivered but nothing arrived at my address", "shipping", "high", "negative"),
    ("Order marked shipped ten days ago and tracking has not moved", "shipping", "high", "negative"),
    ("Courier left the box outside in the rain and it is ruined", "shipping", "high", "negative"),
    ("Delivery went to my old address even though I updated it", "shipping", "high", "negative"),
    ("The package arrived crushed and the contents are broken", "shipping", "high", "negative"),
    ("Half the items are missing from the box I received", "shipping", "high", "negative"),
    ("Signed for delivery I never received, signature is not mine", "shipping", "high", "negative"),
    ("Urgent order for tomorrow has not left the warehouse", "shipping", "high", "negative"),
    ("Tracking number returns nothing on the courier website", "shipping", "medium", "negative"),
    ("Can I change the delivery address before it ships", "shipping", "medium", "neutral"),
    ("Estimated arrival keeps moving further out every day", "shipping", "medium", "negative"),
    ("Need to reschedule delivery for next week, I am away", "shipping", "medium", "neutral"),
    ("Driver could not find the building, how do I add directions", "shipping", "medium", "negative"),
    ("Is it possible to collect from the depot instead", "shipping", "medium", "neutral"),
    ("Two of three parcels arrived, where is the third", "shipping", "medium", "negative"),
    ("Delivery attempted while I was home and nobody knocked", "shipping", "medium", "negative"),
    ("Can the courier leave it with a neighbour", "shipping", "medium", "negative"),
    ("Order shipped to the wrong country entirely", "shipping", "high", "negative"),
    ("How long does standard delivery usually take", "shipping", "low", "neutral"),
    ("Do you ship to the Channel Islands", "shipping", "low", "neutral"),
    ("What courier do you normally use", "shipping", "low", "neutral"),
    ("Can I get a delivery slot on a Saturday", "shipping", "low", "neutral"),
    ("Is there a charge for express shipping", "shipping", "low", "neutral"),
    ("Do you send a notification when the parcel is out for delivery", "shipping", "low", "neutral"),
    ("Will the item come in discreet packaging", "shipping", "low", "neutral"),
    ("Can I combine two orders into one shipment", "shipping", "low", "neutral"),
    ("Returns label was not included in the box", "shipping", "medium", "negative"),
    ("How do I arrange a return collection", "shipping", "medium", "neutral"),
    ("Parcel stuck at customs for two weeks now", "shipping", "high", "negative"),
    ("Consignment shows out for delivery three days running", "shipping", "medium", "negative"),
    ("Wrong item entirely was delivered to me", "shipping", "high", "negative"),
    ("Do you deliver on bank holidays", "shipping", "low", "neutral"),
    ("Need proof of delivery for my insurance claim", "shipping", "medium", "negative"),
    ("Package thrown over the fence and the seal is broken", "shipping", "high", "negative"),
    ("Can I track the order without an account", "shipping", "low", "neutral"),

    # ---------------------------------------------------------------- account
    ("Locked out completely after too many failed attempts", "account", "high", "negative"),
    ("Password reset email never arrives no matter how often I ask", "account", "high", "negative"),
    ("Someone else has logged into my account from another country", "account", "high", "negative"),
    ("Two factor codes are being sent to a phone I no longer own", "account", "high", "negative"),
    ("My account was suspended without any explanation", "account", "high", "negative"),
    ("Cannot sign in since the email address change", "account", "high", "negative"),
    ("Account appears to have been deleted along with all my data", "account", "high", "negative"),
    ("Login loops back to the sign in page every time", "account", "high", "negative"),
    ("Need to transfer ownership to a colleague who is taking over", "account", "medium", "neutral"),
    ("How do I change the email address on file", "account", "medium", "neutral"),
    ("Want to add two more users to the workspace", "account", "medium", "negative"),
    ("Profile picture will not save when I upload it", "account", "medium", "negative"),
    ("Cannot update my phone number in the settings page", "account", "medium", "negative"),
    ("How do I enable two factor authentication", "account", "medium", "neutral"),
    ("Team member left, need to revoke their access", "account", "medium", "negative"),
    ("Session expires far too quickly, logged out constantly", "account", "medium", "negative"),
    ("Notification preferences reset themselves every login", "account", "medium", "negative"),
    ("Where do I download all my data before closing the account", "account", "medium", "neutral"),
    ("How do I permanently close my account", "account", "low", "neutral"),
    ("Can I change my username", "account", "low", "neutral"),
    ("Is there a way to have multiple accounts under one email", "account", "low", "neutral"),
    ("What happens to my data if I stay inactive for a year", "account", "low", "neutral"),
    ("Do you support single sign on with Google", "account", "low", "neutral"),
    ("Can I set the interface language to French", "account", "low", "neutral"),
    ("How many devices can be signed in at once", "account", "low", "neutral"),
    ("Is there an option for a dark theme", "account", "low", "neutral"),
    ("Verification link in the signup email has expired", "account", "medium", "negative"),
    ("Cannot complete registration, it says the email is taken", "account", "high", "negative"),
    ("Recovery codes are not being accepted", "account", "high", "negative"),
    ("Want to merge two accounts I created by mistake", "account", "medium", "negative"),
    ("Role permissions do not seem to apply to the new user", "account", "medium", "negative"),
    ("Do you have an audit log of who signed in and when", "account", "low", "neutral"),
    ("Received a security alert I did not trigger", "account", "high", "negative"),
    ("Cannot remove an old device from the trusted list", "account", "medium", "negative"),
    ("Is there a limit on how many team members I can invite", "account", "low", "neutral"),

    # -------------------------------------------------------------- technical
    ("Application crashes every time I open the reports section", "technical", "high", "negative"),
    ("Site returns a 500 error on every page since this morning", "technical", "high", "negative"),
    ("Export produces an empty file no matter what I select", "technical", "high", "negative"),
    ("Data I saved yesterday has completely disappeared", "technical", "high", "negative"),
    ("Upload fails at ninety percent every single time", "technical", "high", "negative"),
    ("Search returns no results even for things I know exist", "technical", "high", "negative"),
    ("The whole dashboard is blank after the latest update", "technical", "high", "negative"),
    ("Sync between devices stopped working entirely", "technical", "high", "negative"),
    ("Integration webhook has not fired since Tuesday", "technical", "high", "negative"),
    ("API returns 401 with a key that worked last week", "technical", "high", "negative"),
    ("Pages take thirty seconds to load in the afternoon", "technical", "medium", "negative"),
    ("Filters reset whenever I navigate back to the list", "technical", "medium", "negative"),
    ("Charts render incorrectly in Safari but fine in Chrome", "technical", "medium", "negative"),
    ("Attachment preview shows a blank window", "technical", "medium", "negative"),
    ("Notifications arrive hours after the event", "technical", "medium", "negative"),
    ("Keyboard shortcuts stopped responding in the editor", "technical", "medium", "negative"),
    ("Timezone displayed is wrong for users outside the UK", "technical", "medium", "negative"),
    ("CSV import rejects valid rows without saying why", "technical", "medium", "negative"),
    ("Mobile layout overlaps the buttons on smaller screens", "technical", "medium", "negative"),
    ("Print view cuts off the right hand column", "technical", "low", "negative"),
    ("Spelling mistake on the confirmation screen", "technical", "low", "negative"),
    ("Icon for archived items is hard to distinguish", "technical", "low", "negative"),
    ("Would be useful to sort the table by date added", "technical", "low", "negative"),
    ("Tooltip text is cut short on long labels", "technical", "low", "negative"),
    ("Is there a keyboard shortcut for creating a new record", "technical", "low", "neutral"),
    ("Do you have an API rate limit documented anywhere", "technical", "low", "neutral"),
    ("Dark mode contrast is too low to read comfortably", "technical", "low", "negative"),
    ("Feature request, allow bulk editing of tags", "technical", "low", "negative"),
    ("Browser console shows CORS errors on the embed", "technical", "high", "negative"),
    ("Scheduled job has not run for three days", "technical", "high", "negative"),
    ("Report totals do not match the underlying rows", "technical", "high", "negative"),
    ("Autosave overwrote my draft with an older version", "technical", "high", "negative"),
    ("Two users editing the same record lose each other's changes", "technical", "medium", "negative"),
    ("Emails from the system land in spam every time", "technical", "medium", "negative"),
    ("Is there a sandbox environment for testing the API", "technical", "low", "neutral"),

    # ------------------------------------------------------- positive in tone
    # Support queues are not uniformly angry. These carry a real question or
    # request - they are still tickets - but the tone is warm, which is what
    # sentiment is meant to capture. Spread across all four categories so the
    # model cannot learn "positive" as a proxy for one of them.
    ("Great service as always, just need a copy of last month's invoice",
     "billing", "low", "positive"),
    ("Really happy with the plan, thinking of upgrading to the team tier",
     "billing", "low", "positive"),
    ("Thanks for sorting the refund so quickly, much appreciated",
     "billing", "low", "positive"),
    ("Billing portal is much clearer since the redesign, one small question about VAT",
     "billing", "low", "positive"),
    ("Pricing page is refreshingly straightforward, do you offer annual invoicing",
     "billing", "low", "positive"),
    ("Love the product, happy to move to the yearly plan if that helps you",
     "billing", "low", "positive"),
    ("Appreciate the quick credit note, all looks correct now", "billing", "low", "positive"),

    ("Delivery arrived a day early and beautifully packed, thank you",
     "shipping", "low", "positive"),
    ("Courier was excellent, just wondering how to leave feedback for them",
     "shipping", "low", "positive"),
    ("Packaging was completely plastic free which I really appreciated",
     "shipping", "low", "positive"),
    ("Fantastic that you now ship to Ireland, when did that start",
     "shipping", "low", "positive"),
    ("Order came through perfectly, could I set up a repeat delivery",
     "shipping", "low", "positive"),
    ("Really impressed with the tracking updates, very clear", "shipping", "low", "positive"),
    ("Thanks for the fast dispatch, would like to order again this week",
     "shipping", "low", "positive"),

    ("Onboarding was the smoothest I have used, how do I invite my team",
     "account", "low", "positive"),
    ("Two factor setup was painless, thanks for making it simple",
     "account", "low", "positive"),
    ("Enjoying the product a lot, is there a way to add a second workspace",
     "account", "low", "positive"),
    ("Support fixed my login yesterday and were very patient, thank you",
     "account", "low", "positive"),
    ("The new settings layout is a big improvement, one question about roles",
     "account", "low", "positive"),
    ("Single sign on worked first time, delighted", "account", "low", "positive"),
    ("Happy customer here, just checking how to export my data for backup",
     "account", "low", "positive"),

    ("The latest release fixed my export issue, thanks for the quick turnaround",
     "technical", "low", "positive"),
    ("New dashboard is genuinely lovely to use, small suggestion about sorting",
     "technical", "low", "positive"),
    ("API docs are the best I have worked with, one question on rate limits",
     "technical", "low", "positive"),
    ("Performance is noticeably better this week, well done",
     "technical", "low", "positive"),
    ("Dark mode is great, could the contrast be nudged slightly higher",
     "technical", "low", "positive"),
    ("Really useful feature, would love bulk editing added at some point",
     "technical", "low", "positive"),
    ("Integration set up in ten minutes, very impressed", "technical", "low", "positive"),
]


def main() -> None:
    existing = list(csv.DictReader(DATA.open()))
    seen = {row["text"].strip().lower() for row in existing}
    next_id = max(int(row["ticket_id"]) for row in existing) + 1

    added, duplicates = [], 0
    for text, category, priority, sentiment in TICKETS:
        if text.strip().lower() in seen:
            # A duplicate would appear in both train and test after the split,
            # which is the quietest way to inflate an accuracy score.
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