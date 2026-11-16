# Labelling guide: claims in 10-K Risk Factors and MD&A

**Status: frozen 2026-11-06** (owner approved the 13 types, a 25% block sample and the split below). Changing the types now would invalidate the labels.

**Who labelled:** the owner chose (2026-11-06) not to hand-label. The labels in `evals/v3/labels/` were made by Claude (Anthropic), a different model family from the DeepSeek extractor, following this guide, with no human check. Extraction scores therefore measure agreement with an independent model's labels, not accuracy against a human expert.

These labels measure the v3 claim extractor: **precision** (how many extracted claims are real) and **recall** (how many real claims it finds). They also measure the year-to-year change labels (new, changed, unchanged, dropped). Label from the raw text only, without looking at anything the extractor produced.

All examples below come from Micron's and Microsoft's FY2026 10-Ks, which are not in the label set.

## What to label

| Pair | Years | Split |
|---|---|---|
| NVDA | FY2026 and FY2025 | dev |
| AAPL | FY2025 and FY2024 | dev |
| AMAT | FY2025 and FY2024 | test |

**Files:**
- Each filing has a readable pack in `data/v3/label_packs/` (for example `NVDA_2026.md`), and an empty CSV to fill in at `evals/v3/labels/NVDA_2026.csv`.
- The packs split Items 1A and 7 into numbered blocks (`B001`, `B002`, …), each starting at a heading. Label only the blocks marked **LABEL**.
- In the earlier year of each pair, every block's text is shown, so you can find the earlier version of a claim even if its heading was reworded.

**Check your work:** `edgar label-check` checks every file. It confirms that each quote is copied verbatim, that types are valid, and that links resolve.

## What counts as a claim

A claim is **one specific assertion** in Item 1A or Item 7 that does both of these:

1. **Fits one of the 13 types below.** If it fits none, skip it.
2. **Has at least one concrete anchor:** a named entity (company, country, regulation, product, segment), a number, percentage or date, or a specific event. Generic risk language with no anchor is **not** a claim.

| Label it | Skip it |
|---|---|
| "…approximately 70% of our revenue in 2026 was from products shipped to customer locations outside the United States." (a number) | "The semiconductor memory and storage markets are highly competitive." (no anchor) |
| "Additionally, we have a long-term strategic partnership with OpenAI." (a named partner) | "Our AI strategy also depends in part on strategic relationships with third parties…" (no one is named) |
| "…the CAC determined that critical information infrastructure operators in China may not purchase Micron products…" (a named regulator, country and event) | "…compliance with trade, technical standards, and other laws in a variety of jurisdictions;" (no specifics) |

**Granularity:**
- One paragraph can hold several claims, of the same type or different types. Label each separately.
- If the same assertion is repeated within a section, label it **once**, at its first occurrence.
- If the same assertion appears in both Item 1A and Item 7, label it in **each**: sections are labelled separately.
- A heading that only announces a topic ("A significant portion of our revenue is concentrated with certain customers and end markets.") is not a claim on its own. The specific sentence under it is.
- **Lists:** if a bullet list names several countries or companies for one point, it's one claim, with all of them in `entities`. If the bullets make different points, label them separately.

**Hypotheticals:**
- Label disclosures of what *is*: facts, current dependencies, events that happened, and named exposures.
- Skip pure "could" or "may" statements unless they name a specific entity, regulation or event. "We may be subject to export restrictions" is skipped. "The U.S. government's export restrictions on our A100 and H100 products…" would be labelled.

## Claim types

| `claim_type` | What it covers | Example (Micron or Microsoft FY2026) |
|---|---|---|
| `customer_concentration` | Reliance on particular or few customers: a named customer, or a share of revenue from a customer or group | (a named customer or a "% of revenue" statement) |
| `supplier_dependence` | Reliance on particular or few suppliers, foundries, or sole/single sources | "…in some cases, materials, components, or services are provided by a single or sole source…" |
| `export_controls_trade` | Export restrictions, entity lists, tariffs, sanctions, trade actions | "…the CAC determined that critical information infrastructure operators in China may not purchase Micron products…" |
| `geographic_exposure` | Revenue, operations or supply concentrated in a named country or region | "…approximately 70% of our revenue in 2026 was from products shipped to customer locations outside the United States." |
| `segment_change` | The reportable segments, or a change to them | "We report our financial performance based on the following three segments: Productivity and Business Processes, Intelligent Cloud, and More Personal Computing." |
| `strategic_partnership` | A named partner relationship (investment, joint development, alliance) | "Additionally, we have a long-term strategic partnership with OpenAI." |
| `acquisition_divestiture` | A named acquisition, merger, divestiture or spin-off, completed or pending | "For example, in October 2023 we completed our acquisition of Activision Blizzard, Inc." |
| `capacity_investment` | Fabs, datacenters, capacity expansion, capex plans, and government incentives for them | "…other government incentives, including through the CHIPS Act…" |
| `restructuring` | A specific restructuring plan, layoffs or facility closures, with charges or scope | (a plan with a date, charge or headcount) |
| `legal_regulatory` | A named investigation, lawsuit, regulator action or regulation that applies to the company | (a named case, agency or regulation) |
| `cybersecurity_incident` | A specific security incident that happened (not general cyber risk) | "…a nation-state associated threat actor used a password spray attack to compromise a legacy test account…" |
| `ai_demand` | AI as a driver of demand, revenue, investment or risk | "AI currently is, and is expected to continue to be, a significant demand driver for our products." |
| `competition` | Named competitors, or a specific competitive development | (a list of named competitors) |

**When two types fit:**
- Pick the one the sentence is mainly *about*, and note the other in `notes`.
- **Trade actions against a country:** `export_controls_trade` wins over `geographic_exposure`, because the action is the point.
- **A named customer's share of revenue:** `customer_concentration`.

There is deliberately **no "other" type**. Claims outside these 13 are skipped. This keeps precision and recall measurable and the labelling bounded.

## Filling in the CSV

One row per claim. Spreadsheets handle the CSV quoting of commas inside quotes.

| Column | What to write |
|---|---|
| `claim_id` | `TICKER-FY-NNN`, numbered in reading order: `NVDA-2026-001`, `NVDA-2026-002`, … |
| `ticker`, `fiscal_year` | e.g. `NVDA`, `2026` |
| `section` | `1A` or `7` |
| `block` | The block number from the pack, e.g. `17` for B017 |
| `claim_type` | One of the 13 types, exactly as written above |
| `entities` | Named things, as `kind:Name` separated by `;`. Kinds: company, country, regulation, regulator, product, segment, other. Example: `regulator:CAC;country:China` |
| `statement` | One line in your own words, e.g. "China's CAC barred critical-infrastructure operators from buying Micron products (May 2023)" |
| `quote` | Copied **verbatim** from the pack: the shortest span (usually one sentence, at most three) that makes the claim. Copy and paste; don't retype. |
| `prior_claim_id` | Later year only: the id of the same claim in the earlier year, or blank if new |
| `change` | Later year only: `new`, `changed` or `unchanged` |
| `notes` | Anything unsure, a second type that also fits, or why a link is unusual |

## Linking the two years

Label the **earlier year first**, then the later year. For each later-year claim, find the same claim in the earlier year.

**"The same claim"** means the same type, the same subject (the same customer, regulation, country or segment) and the same underlying assertion. The wording or numbers may differ.
- "One customer was 13% of revenue" and, a year later, "one customer was 22% of revenue" is the same claim, **changed**.

| `change` | When |
|---|---|
| `unchanged` | Same claim, same wording (punctuation and spacing aside). Pick the same quote span in both years. |
| `changed` | Same claim, but the wording, numbers, dates or listed entities differ |
| `new` | No earlier version anywhere in that year's Item 1A or 7. Search the whole earlier pack, including `(skip)` blocks. |

**Dropped** is never labelled directly. An earlier-year claim that no later-year claim links to counts as dropped.

**When the earlier version is in a `(skip)` block:**
- Add it to the earlier-year CSV with that block number, and write `link target` in `notes`.
- The checker accepts those rows only if a later-year claim links to them.
- They are used for the change labels, not for scoring the extractor.

**More rules:**
- **Each earlier claim links to at most one later claim.** If one earlier claim was split into two, link the closer one and mark the other `new`, with a note.
- **Merged claims:** if two earlier claims were merged into one, link the closer one. The other earlier claim then counts as dropped; say so in `notes`.
- **Format-only changes:** if the change is purely a format change (the same text moved under a new heading, or a list reflowed), it's still `unchanged` if the words match, and `changed` if they don't. Add `format only` to `notes`, so format churn can be counted separately.

## Worked example

From Micron FY2026 Item 1A, labelled as if it were the later year (block numbers and FY2025 ids are illustrative):

| claim_id | section | block | claim_type | entities | quote | prior_claim_id | change |
|---|---|---|---|---|---|---|---|
| MU-2026-014 | 1A | 9 | geographic_exposure | country:United States | Further, in 2026, approximately one-third of our revenue was from sales to customers who have headquarters located outside the United States, while approximately 70% of our revenue in 2026 was from products shipped to customer locations outside the United States. | MU-2025-012 | changed |
| MU-2026-031 | 1A | 24 | export_controls_trade | regulator:CAC;country:China;company:Micron | Following the May 2023 decision of its cybersecurity review of our products sold in China, the CAC determined that critical information infrastructure operators in China may not purchase Micron products, impacting our revenue with companies headquartered in mainland China and Hong Kong, including direct sales, as well as indirect sales through distributors. | MU-2025-029 | unchanged |

The first row is `changed` because the percentages and year differ from FY2025's sentence. The second is `unchanged` if FY2025 has the same sentence word for word.

## Effort

Measured on the packs at the agreed 25% block sample (seed 3):

| Filing | Blocks to label | Words to read | Words in full |
|---|---|---|---|
| NVDA FY2026 | 15 of 60 | 6,365 | 22,315 |
| NVDA FY2025 | 11 of 65 | 4,358 | 22,372 |
| AAPL FY2025 | 12 of 47 | 3,011 | 12,454 |
| AAPL FY2024 | 10 of 49 | 2,555 | 12,222 |
| AMAT FY2025 | 12 of 49 | 4,594 | 16,100 |
| AMAT FY2024 | 10 of 50 | 4,075 | 16,493 |
| **Total** | **70 of 320** | **24,958 (roughly 3 hours by hand)** | 101,956 |

The earlier-year packs also show the (skip) blocks, for finding link targets. You only search those; you don't read them in full.

## FAQ

Questions are added here as they come up while labelling.

- **A sentence has a number but it's a general industry figure, not about the company.** Skip it unless it's tied to the company's own exposure.
- **The claim is a table, not a sentence.** Quote the table row as it appears in the pack (one line), and note `table` in `notes`.
