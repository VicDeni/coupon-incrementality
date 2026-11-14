# Memo: what our coupons really earn

**To:** VP Marketing  **From:** Vic, Analytics  **Re:** Coupon program return and next steps

## The short version

Our reporting says coupon campaigns return about **$43 of gross profit for every $1** we spend on them. Our best estimate of the real figure is about **$4.60**, and we cannot rule out that the program only breaks even.

Nine-tenths of what the dashboard shows is not something the coupons did. We mail coupons to our best customers, and our best customers spend more with or without a coupon.

I recommend four things:

1. **Change how we keep score.** Hold back a random 5–10% of every mailing list and compare. It costs almost nothing, and it is the only way to turn "probably profitable" into a number nobody can argue with.
2. **Keep the personalised campaigns while we test them.** They are where most of the estimated return comes from.
3. **Put the same-for-everyone product coupons on probation.** On the products they cover, they do not earn back their mailing cost.
4. **Do not invest in smarter targeting yet.** We found no reliable sign that a model can pick the households that respond best.

## What we measured

One year of purchases from about 2,500 households, covering 22 campaigns and 5,898 mailings. The campaigns cost roughly $5,400: coupons redeemed plus an assumed 50 cents to print and mail each piece.

| | Gross profit | Per $1 spent |
|---|---|---|
| What the dashboard would report | $231,000 | $42.67 |
| Best estimate of what the coupons actually added | $25,000 | $4.64 |
| Range we cannot rule out | −$900 to $51,000 | −$0.16 to $9.45 |

The gap is not an accounting error. Households that received coupons spend $53 a week; households that did not spend $18. The dashboard credits that whole difference to the coupon. When we compare each recipient with similar shoppers who did not get the mailing, the coupon adds about $2 a week while a campaign runs, a lift of 1% to 5% depending on the method, not 188%.

We reached that conclusion three different ways, each resting on a different assumption. They disagree on the exact figure and agree on the size: single digits.

## The two kinds of campaign are not alike

**Personalised campaigns** (16 coupons chosen from each household's own purchase history) return an estimated **$5.71 per $1**. Every method we tried lands between $2.60 and $5.70 per $1, and every one also leaves break-even inside its margin of error. This part of the program is probably profitable, but this data cannot prove it.

**Same-for-everyone campaigns** (one set of product coupons mailed to a list) raised sales of the couponed products by about 3%. That is 15 cents per household, against 68 cents of cost. They pay back only if they also bring people into the store to buy other things. Our best estimate of that store-wide effect is positive, about $6 per household, but it is too uncertain to bank: the range runs from −$1 to +$14.

Two worries turned out not to be problems. Shoppers did **not** stock up and then buy less afterwards. Coupons also did **not** simply pull sales from neighbouring products on the shelf.

## Who should get coupons

We looked hard for households that respond more than others and did not find them. A model that ranked households by expected response did no better than chance when tested on households it had not seen.

We also compared five ways of choosing who gets the same number of mailings. None is a proven improvement on our current list, and none is clearly profitable on its own at this sample size. The estimates lean towards mailing *more* households rather than different ones. The accompanying simulator shows expected profit for any budget and margin; treat its curves as hypotheses to test.

## Risks and what would change this advice

- **The margin and mailing cost are assumptions.** We used a 27.5% gross margin and 50 cents per piece. At $1.00 per piece the best estimate falls to $3.00; at 25 cents it rises to $6.40. The conclusion that the dashboard overstates the return about tenfold holds across that range.
- **This is one year at one retailer and 2,500 households.** That is enough to show the true return is a small fraction of the reported one. It is not enough to show it is above break-even, or to rank households or campaigns.
- **We could not randomise.** These estimates depend on having captured what drove the mailing lists. A random holdout removes that dependence, which is why it is recommendation number one.
- **We do not know who funds each coupon.** If manufacturers reimburse the face value, our own return is higher than stated.

## Next steps

| When | Action |
|---|---|
| Next campaign | Hold out a random 10% of the list. Read the result four weeks after the campaign ends. |
| Next quarter | Run one same-for-everyone campaign with a holdout before committing to the next. |
| Next quarter | Test extending a personalised campaign to more households, with a holdout. |
| Ongoing | Report coupons on incremental profit per $1, not redemptions. |

*The method, the checks and everything that went wrong along the way are in the project README.*
