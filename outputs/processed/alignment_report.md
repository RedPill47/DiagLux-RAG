# Alignment and Validation Report

Generated: 2026-06-12T23:48:00+00:00  
Data root: `C:\Users\hedit\DiagLux-RAG\dataset`  
Questions: 640  |  Global shuffle seed: 13

## Per-text title match

| text_id | title | KB title match | questions |
|---|---|---|---|
| text1 | Catherine, ech sinn esou glécklech | OK | 40 |
| text2 | De Mathematik-Proff | OK | 40 |
| text3 | De Pablo an d’Juliette | OK | 40 |
| text4 | Dräizéng | OK | 40 |
| text5 | Blues | OK | 40 |
| text6 | Ech denken nach vill un de Mike | OK | 40 |
| text7 | Meng éischt Zäit am Lycée | OK | 40 |
| text8 | Hausaufgaben – derfir, dergéint, oder wéi? | OK | 40 |
| text9 | Den Ersatzschoulmeeschter | OK | 40 |
| text10 | Schoulliewen am Krich | OK | 40 |
| text11 | De Kunibert vun Hesper | OK | 40 |
| text12 | De Siegfried an d’Melusina | OK | 40 |
| text13 | Poker | OK | 40 |
| text14 | D’Schnurreli | OK | 40 |
| text15 | E Muerd am Gréngewald | OK | 40 |
| text16 | Vakanzën | OK | 40 |

## Span alignment status counts

| status | criticalSpan | distractorSpan |
|---|---|---|
| exact | 594 | 625 |
| dehyphen | 7 | 3 |
| fuzzy | 34 | 6 |
| multiple | 0 | 1 |
| unresolved | 0 | 0 |
| empty | 5 | 5 |

Located (exact + dehyphen + fuzzy + multiple + in-title): critical 635/640, distractor 635/640.

Of the fuzzy spans, 22 critical and 2 distractor spans carry `"partial": true`: the annotated span concatenates non-contiguous passages of the text, and the recorded offsets cover only the longest sentence piece that aligned uniquely. `partial` is an additive extension of the span schema (consumers ignore unknown keys).

## Question counts per cognitive type

| cognitive type | questions |
|---|---|
| Retrieve | 128 |
| Interpret | 192 |
| Inferential | 192 |
| Evaluative | 128 |

## Unresolved spans

None — every span was located (or is empty/in-title).

## Chunking

| strategy | total chunks | per text |
|---|---|---|
| paragraph | 187 | text1:12, text2:15, text3:10, text4:7, text5:4, text6:6, text7:15, text8:7, text9:7, text10:8, text11:12, text12:11, text13:26, text14:17, text15:23, text16:7 |
| overlap | 213 | text1:15, text2:15, text3:11, text4:7, text5:5, text6:7, text7:17, text8:6, text9:11, text10:11, text11:13, text12:9, text13:29, text14:18, text15:25, text16:14 |
| sentence | 1351 | text1:74, text2:106, text3:83, text4:93, text5:28, text6:68, text7:76, text8:42, text9:48, text10:65, text11:69, text12:82, text13:180, text14:86, text15:186, text16:65 |

### Overlap coverage check (union of spans must equal each full body)

- `text1` overlap: 15 chunks, union covers [0, 6260) — OK
- `text2` overlap: 15 chunks, union covers [0, 7072) — OK
- `text3` overlap: 11 chunks, union covers [0, 4672) — OK
- `text4` overlap: 7 chunks, union covers [0, 3054) — OK
- `text5` overlap: 5 chunks, union covers [0, 2049) — OK
- `text6` overlap: 7 chunks, union covers [0, 3090) — OK
- `text7` overlap: 17 chunks, union covers [0, 7650) — OK
- `text8` overlap: 6 chunks, union covers [0, 3110) — OK
- `text9` overlap: 11 chunks, union covers [0, 5092) — OK
- `text10` overlap: 11 chunks, union covers [0, 5525) — OK
- `text11` overlap: 13 chunks, union covers [0, 5480) — OK
- `text12` overlap: 9 chunks, union covers [0, 4159) — OK
- `text13` overlap: 29 chunks, union covers [0, 12174) — OK
- `text14` overlap: 18 chunks, union covers [0, 7810) — OK
- `text15` overlap: 25 chunks, union covers [0, 10985) — OK
- `text16` overlap: 14 chunks, union covers [0, 6206) — OK

Note: the clean texts contain no blank lines (hard-wrapped single-newline layout), so the `paragraph` strategy groups consecutive lines into natural units flushed at sentence-final line ends (>= 60 tokens, hard cap 180).
