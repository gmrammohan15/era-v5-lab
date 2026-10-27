# Medicine sheet: ind-hi-hplt (gmrammohan15)

Write one section for each disease that you found more than five times.

**Status (27 Oct):** built from the first 150 of 500 labelled rows. Counts, examples and numbers will be updated
with all 500 rows by Sat 31 Oct. Sections are ordered by count.

**How the numbers were measured.** Every medicine was run on the 150 labelled documents and on the whole pack (500).
The word lexicon and the site lists that some medicines use were built leave-one-out: a document is never cured with
statistics that include itself.
- **cures**: labelled documents with the disease that the medicine changes or drops.
- **side effects**: labelled documents *without* the disease that it changes or drops.
- **clean touched**: documents I reviewed as `clean` that it changes or drops (target 0).

**Harm test:** the Hindi gold set (`teams/02-data-b/gold/`, due 29 Oct) is not published yet, so every harm test below
is **pending**. The runner is ready: it applies each medicine to the gold documents and fails a medicine that changes or
drops more than 1% of them. Until then the "clean touched" count on my reviewed rows stands in for it: it is 0 for every medicine.

Shared definitions used by the code below:

```python
import re
DEVA, CONS, MATRA = "\u0900-\u097F", "\u0915-\u0939", "\u093E-\u094C"
LETTER = "\u0900-\u0963\u0966-\u097F"          # Devanagari letters and signs, without । and ॥
WORD_RE = re.compile(f"[{DEVA}]+")
norm_line = lambda line: re.sub(r"\s+", " ", line).strip()
```

## Medicine 1: C1 boilerplate

1. **Disease:** C1, 71 of 150 documents (40 fix, 31 drop). Examples:
   - `f6f77aeff3b688d0`: `यह भी पढ़ें` at offset 19713 (related-article teaser).
   - `57294dfeff306304`: `कॉपीराइट एससी मनोवैज्ञानिक उद्यम लिमिटेड` at offset 8538 (copyright footer).
   - `7261baa0a3edfb1d`: `No comments:` at offset 8983 (Blogger comment widget).
2. **Medicine:** remove template lines that appear in 3+ documents (site menus, footers), and short lines (60 chars
   or less) that contain a boilerplate phrase. From the first such line in the last 30% of the document, cut the whole
   tail ("Related posts", comment form, rating widget, share buttons).
   ```python
   C1_PHRASES = [
       "यह भी पढ़ें", "ये भी पढ़ें", "इसे भी पढ़ें", "यह भी पढे", "ये भी पढे", "और पढ़ें", "आगे पढ़ें", "पूरा पढ़ें",
       "संबंधित खबर", "सम्बंधित खबर", "संबंधित लेख", "संबंधित पोस्ट", "लेटेस्ट खबर", "ताज़ा खबर", "ताजा खबर",
       "शेयर करें", "शेयर करे", "लाइक करें", "फॉलो करें", "सब्सक्राइब", "डाउनलोड करें", "हमसे जुड़ें", "हमें फॉलो",
       "कमेंट", "टिप्पणी करें", "प्रातिक्रिया दे", "प्रतिक्रिया दें", "अपनी राय", "जवाब दें", "रद्द करें",
       "सर्वाधिकार सुरक्षित", "सभी अधिकार सुरक्षित", "कॉपीराइट", "विज्ञापन", "गोपनीयता नीति", "अस्वीकरण", "संपर्क करें",
       "हमारे बारे में", "Share on", "WhatsApp", "Facebook", "Twitter", "Telegram", "Instagram", "YouTube", "Pinterest",
       "LinkedIn", "Read More", "Read more", "Click here", "Subscribe", "Follow us", "Advertisement", "Copyright", "©",
       "All Rights Reserved", "All rights reserved", "Privacy Policy", "Disclaimer", "Terms of", "Cookie", "cookies",
       "Tags:", "Tags :", "Posted in", "Filed under", "Leave a Reply", "Leave a Comment", "Comments", "Next Post",
       "Previous Post", "Related Posts", "Home »", "Home >", "Skip to content", "Menu"]
   C1_RE = re.compile("|".join(map(re.escape, sorted(C1_PHRASES, key=len, reverse=True))))

   def cure_c1(text, shared_lines):        # shared_lines: lines that occur in 3+ documents of the site/pack
       out, pos, n = [], 0, len(text)
       for line in text.split("\n"):
           nl = norm_line(line)
           bp = bool(nl) and (nl in shared_lines or (len(nl) <= 60 and C1_RE.search(nl)))
           if bp and pos > 0.7 * n:
               break                           # boilerplate tail: cut everything after it
           if not bp:
               out.append(line)
           pos += len(line) + 1
       return "\n".join(out).rstrip()
   ```
   Example (`ca38e3770cd4bd18`, offset 8157): `…भिन्न हो सकते हैं।⏎अधिकतम रेटिंग 5⏎न्यूनतम स्कोर 4⏎रेटिंग की संख्या 915…` becomes `…भिन्न हो सकते हैं।`
3. **Action:** fix the text.
4. **Harm test:** pending (gold set). Pack check: cures 42 of 71; side effects 0 of 79; clean touched 0. Changes 134 of 500 pack documents.
5. **Side effects:** misses boilerplate that is long, unique to one page, or written as a normal sentence (29 of 71,
   e.g. `फारवर्ड प्रेस वेब पोर्टल के अतिरिक्त…` in `157a92dae8c7615d`). A short content line that names a platform
   ("Facebook ने कहा…") in the last 30% would cut the tail early; no case in the 150 rows. The shared-line rule needs
   site-level statistics at corpus scale, not pack-level.

## Medicine 2: D4 machine translation

1. **Disease:** D4, 35 of 150 documents (30 drop, 5 fix). Examples:
   - `a85a2363de961b2e`: `मैं अपने हृदय की गहराई से उसे धन्यवाद दिये बिना रह पाया` at offset 23753 (negation lost: "could not help thanking" became "could help thanking").
   - `ca38e3770cd4bd18`: `भी ianbolshuyu मूल्य किग्रा` at offset 2449 (untranslated Russian transliteration inside Hindi).
   - `7a2bc38b934a51ad`: `तेजी से dorozhanie Bitcoin` at offset 4146 (same).
2. **Medicine:** there is no text repair for wrong meaning. Drop documents from sites whose reviewed pages are
   machine-translated, and stitched spam pages (detected by the spam rule shared with C4 and C5).
   ```python
   SPAM_TERMS = ["विदेशी मुद्रा", "द्विआधारी विकल्प", "बाइनरी विकल्प", "बिटकॉइन", "पैसे कमा", "PAMM", "Forex",
                 "forex", "binary option", "Bitcoin"]

   def is_spam(text):           # only spam-specific terms: निवेश or ट्रेडिंग also occur in real business news
       hits = {t: text.count(t) for t in SPAM_TERMS if t in text}
       return sum(hits.values()) >= 4 and len(hits) >= 2

   def cure_d4(text, url, mt_sites):   # mt_sites: sites with reviewed D4 drops (healthcarestudies.in, onlinestudies.com, ...)
       return None if urlparse(url).netloc in mt_sites or is_spam(text) else text
   ```
3. **Action:** drop the document.
4. **Harm test:** pending (gold set). Pack check (leave-one-out site list): cures 23 of 35; side effects 0 of 115; clean touched 0.
5. **Side effects:** misses machine translation on sites seen only once (12 of 35), and the 5 D4 documents kept as fix
   (translated sermons with a few wrong-sense sentences) are not touched, by design. At corpus scale the site list
   must be replaced or backed by a model-based detector (e.g. a classifier on round-trip translation similarity or
   perplexity), because a site list never generalises to new sites.

## Medicine 3: A5 legacy-font conversion damage

1. **Disease:** A5, 32 of 150 documents. Text typed in Krutidev/Chanakya fonts and converted to Unicode. Examples:
   - `a02caf8e245bbcf3`: `परेषानियों` at offset 326 (`ष` written for `श`).
   - `1dc03c85e4d5e1db`: `चित्राें` at offset 27 (`ा` + `ें` for `ों`).
   - `3c93f63e4f149867`: `देष` at offset 122; `e96fe1f3108f353f`: `ऋऊख` at offset 7780.
2. **Medicine:** fixed character rules, then a word-level rule for `ष` / `श`. A blind `ष` to `श` replacement is
   wrong (`राष्ट्र`, `कृषि`, `क्षेत्र` are correct), so a word is switched only when the `श` spelling is clearly the
   common word in clean Hindi text.
   ```python
   A5_RULES = [
       (re.compile("र्इ"), "ई"),
       (re.compile("ध्द"), "द्ध"),                                # बौध्दिक -> बौद्धिक
       (re.compile("द्बारा"), "द्वारा"),
       (re.compile("\u093C\u093C+"), "\u093C"),                   # double nukta
       (re.compile(f"([डढ])([{MATRA}])\u093C"), "\\1\u093C\\2"),  # बडे़ -> बड़े
       (re.compile(f"([डढ])([कखगनलमतरसद])\u093C"), "\\1\u093C\\2"),  # सडक़ -> सड़क, लडऩे -> लड़ने
       (re.compile("\u093E\u0947"), "\u094B"),                    # ाे -> ो
       (re.compile("\u093E\u0948"), "\u094C"),                    # ाै -> ौ
       (re.compile(f"([{MATRA}])\\1"), "\\1"),                    # केे -> के
   ]

   def fix_sha(word, lexicon):         # lexicon: Counter of word frequencies in clean Hindi text
       if "ष" not in word:
           return word
       best, best_n = word, lexicon[word]
       pos = [i for i, c in enumerate(word) if c == "ष"]
       for mask in range(1, 2 ** len(pos)):
           alt = "".join("श" if i in pos and mask >> pos.index(i) & 1 else c for i, c in enumerate(word))
           if lexicon[alt] > best_n:
               best, best_n = alt, lexicon[alt]
       return best if best_n >= 3 and best_n > 3 * lexicon[word] else word

   def cure_a5(text, lexicon):
       for rx, rep in A5_RULES:
           text = rx.sub(rep, text)
       return WORD_RE.sub(lambda m: fix_sha(m.group(), lexicon), text)
   ```
   Examples: `परेषानियों` → `परेशानियों`, `चित्राें` → `चित्रों`, `देष` → `देश` (while `राष्ट्र` in the same sentence stays).
   In `3c93f63e4f149867` the `ष` rule corrects 61 words (`प्रकाष`, `अधिषासी`, `प्रदर्षनी`, `विषेष` → `विशेष`) and leaves every correct `ष` word alone.
3. **Action:** fix the text.
4. **Harm test:** pending (gold set). Pack check: cures 19 of 32; side effects 4 of 118, all four correct cures of
   damage the labels missed (`आवष्यकताओं`, `दृष्य`, `स्थाानों`, `ऋषिकेष`); clean touched 0.
5. **Side effects:** does not repair a misplaced reph (`सामथ्र्य`, `दुव्र्यहार`, `पाट्र्स`) or Latin junk left by
   the converter (`Ÿ` in `प्रवृŸिागत`, `ª` in `टैªनिंग`, `÷`); these need a full Krutidev reverse map and are the
   13 uncured documents. The `ष` rule can miss rare words (`निर्देषानुसार`) and could wrongly switch a rare correct
   `ष` word whose `श` twin is common; it needs a large clean lexicon (pack-level counts are small).

## Medicine 4: D3 natural code-mixing

1. **Disease:** D3, 29 of 150 documents. Examples:
   - `f6f77aeff3b688d0`: `back to the basics, back to the nature` at offset 5707 (an English phrase quoted in a Hindi article).
   - `57294dfeff306304`: `spaying और neutering` at offset 4622.
   - `84b3adb7174dfd68`: `youtube पर videos upload करके` at offset 5424.
2. **Medicine:** none. Hindi-English mixing is how Hindi is written online, and the model must learn it. Per the
   reviewer policy, D3 alone means `clean`. English boilerplate (share buttons, SEO headings) is removed by the C1
   medicine, not here.
   ```python
   def cure_d3(text):
       return text
   ```
3. **Action:** keep (no change).
4. **Harm test:** pending (gold set); the medicine changes nothing, so it changes 0 gold documents. Pack check: 0 changes.
5. **Side effects:** none from the medicine. The risk is in labelling: an English run that is really scraped junk
   must be coded C1 or C5, not D3.

## Medicine 5: C4 two documents joined

1. **Disease:** C4, 16 of 150 documents (13 drop, 3 fix). Examples:
   - `ca38e3770cd4bd18`: `ज्यादातर मामलों में पुलिस ने 32 बोर पिस्तौल` at offset 957 (a crime story inside a forex page).
   - `8fed2d4464e4cc06`: `आज आपका दिन मिला जुआ असर देने वाला रहेगा` at offset 3368 (a horoscope inside a forex page).
   - `84b3adb7174dfd68`: `वायुयान का नोदक नोदक या प्रोपेलर` at offset 7846.
2. **Medicine:** in this pack, joined documents are almost always spam pages stitched from unrelated snippets. Drop
   those with the spam rule (Medicine 2). For the rest there is no safe split point, so keep them.
   ```python
   def cure_c4(text):
       return None if is_spam(text) else text
   ```
3. **Action:** drop the document (stitched spam); otherwise keep.
4. **Harm test:** pending (gold set). Pack check: cures 11 of 16; side effects 0 of 134; clean touched 0.
5. **Side effects:** misses joined news round-ups (several datelined items on one page, e.g. `3c93f63e4f149867`) and
   non-spam joins; those stay in the data. A splitter on datelines (`नई दिल्ली।`, `(वार्ता)`) could separate news round-ups later.

## Medicine 6: G4 fake danda

1. **Disease:** G4, 14 of 150 documents. ASCII `|`, Latin `I` or `l`, or `II` typed instead of `।` / `॥`. Examples:
   - `f6f77aeff3b688d0`: `नहीं मिला|` at offset 6860.
   - `7952deb7a1c71995`: `सोचने को विवश कर सकती है|` at offset 8222.
   - `4b82ad7e92695e55`: `किया हैl` at offset 14803 (Latin `l`, about 300 times in this document).
2. **Medicine:** replace the fake mark only where it really ends a sentence: after a Hindi letter (or closing quote),
   at least 25 characters after the previous `।`, `|` or line break, and followed by Hindi or a line end. Title and
   menu separators (`शीर्षक | साइट`, `विज्ञान | गणित`) and English (`मैं I am`) do not match.
   ```python
   G4_FAKE = re.compile(f"(?<=[{LETTER}\"'”’])\\s?(II|\\|\\||[Il|])"
                        f"(?=[ \\t]+[\u0904-\u0939\u0958-\u0961\"'“‘]|[ \\t]*\\n|[ \\t]*$)")

   def cure_g4(text):
       def rep(m):
           start = max(text.rfind(ch, 0, m.start()) for ch in "।|\n")
           if m.start() - start - 1 < 25:
               return m.group()                       # short segment: a separator, not a sentence end
           return "॥" if len(m.group(1)) == 2 else "।"
       return G4_FAKE.sub(rep, text)
   ```
   Example (`f6f77aeff3b688d0`): `मालिक नहीं मिला| नौकर से` → `मालिक नहीं मिला। नौकर से`.
3. **Action:** fix the text.
4. **Harm test:** pending (gold set). Pack check: cures 13 of 14; side effects 6 of 136, all in documents already
   dropped for other reasons; clean touched 0. One of the six is a correct cure the labels missed (`हैं I इसके`).
5. **Side effects:** a Roman numeral after Hindi is taken for a danda (`एनसीएए डिवीजन I सॉकर` → `डिवीजन। सॉकर`;
   "World War I" written in Hindi has the same risk). Long menu items separated by `|` can be converted. Misses
   `*` used as a danda (`91ad466ec2dd9ef4`, the 1 uncured document) and lowercase `i`.

## Medicine 7: D5 translationese

1. **Disease:** D5, 12 of 150 documents. Grammatical Hindi with English sentence structure, mostly translated sermons. Examples:
   - `a85a2363de961b2e`: `को दिखने में एक इंसान जैसा बनाया` at offset 1762.
   - `7261baa0a3edfb1d`: `को धन्यवाद करती हैं` at offset 4573 (English "thank someone" pattern).
   - `d25f5eb32b944379`: `मैंने अपनी राय को जन्म दिया` at offset 8196 ("gave birth to my opinion").
2. **Medicine:** no text repair (rewriting style needs a model). Keep the document, and tag it so the data mix can
   down-weight translated text. The tag uses sites whose reviewed pages are translationese.
   ```python
   def flag_d5(url, translated_sites):    # e.g. hi.godfootsteps.org, hi.easternlightning.org
       return urlparse(url).netloc in translated_sites
   ```
3. **Action:** fix (keep, tag `translated`), per the reviewer policy that D5 alone is `fix`.
4. **Harm test:** pending (gold set); the medicine never changes or drops text, so it changes 0 gold documents. Pack check: tags 5 of 12; tags 0 others.
5. **Side effects:** misses translationese on new sites (7 of 12). A native page hosted on a translation site would
   be tagged. At corpus scale a classifier (translated vs native Hindi) should replace the site list.

## Medicine 8: F1 personal data

1. **Disease:** F1, 12 of 150 documents. Examples (type and position only; values are never copied):
   - `157a92dae8c7615d`: phone at offset 9777 (an author's mobile in a contact line).
   - `4b82ad7e92695e55`: email at offset 28595 (a private person's contact block).
   - `3c8e378530fce8ba`: vehicle registration number at offset 145.
2. **Medicine:** replace each value with a typed placeholder (`<PHONE>`, `<EMAIL>`, `<LANDLINE>`, `<AADHAAR>`, `<PAN>`).
   ```python
   EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
   PHONE_RE = re.compile(r"(?<![\d])(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?![\d])|\+\d{1,3}[\s-]?\d(?:[\s-]?\d){6,13}(?![\d])")
   LANDLINE_RE = re.compile(r"(?<![\d])0\d{2,4}[\s-]\d{6,8}(?![\d])|(?<![\d])0\d{2,4}[\s-]\d{3,4}[\s-]\d{3,4}(?![\d])"
                            r"|\(\+?\d{1,5}\)\s?\d{3,8}[\s-]?\d{3,8}(?![\d])")
   OTHER_PHONE_RE = re.compile(r"(?<![\d-])(?:\d{5}-\d{5}|\d{2} \d{2} \d{3} \d{4}|1800[ -]?\d{3}[ -]?\d{3,4})(?![\d])")
   AADHAAR_RE = re.compile(r"(?<![\d])\d{4}\s\d{4}\s\d{4}(?![\d])")
   PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
   F1 = [("<EMAIL>", EMAIL_RE), ("<PHONE>", PHONE_RE), ("<LANDLINE>", LANDLINE_RE), ("<PHONE>", OTHER_PHONE_RE),
         ("<AADHAAR>", AADHAAR_RE), ("<PAN>", PAN_RE)]

   def cure_f1(text):
       for tag, rx in F1:
           text = rx.sub(tag, text)
       return text
   ```
   Example (`157a92dae8c7615d`, offset 9777): `मोबाइल : <value>, ईमेल : <value>` → `मोबाइल : <PHONE>, ईमेल : <EMAIL>`.
3. **Action:** fix the text.
4. **Harm test:** pending (gold set). Pack check: cures 9 of 12; side effects 1 of 138; clean touched 0.
5. **Side effects:** masks number ranges that look like phones (a rouble price range in `1f6c20e8fe2546b7`: the 1
   side effect) and public organisation helplines (also masked; acceptable). Misses vehicle numbers, home addresses
   and names (3 uncured documents); those need a named-entity model, and a news report naming an accused person is
   public record, not F1.

## Medicine 9: C5 keyword stuffing and link lists

1. **Disease:** C5, 12 of 150 documents (11 drop, 1 fix). Examples:
   - `ca38e3770cd4bd18`: `विदेशी मुद्रा पर पैसे कमाएं` at offset 561.
   - `7a2bc38b934a51ad`: `दृष्टिकोण कंप्यूटर का उपयोग कर बिटकॉइन कैसे कमाएं है` at offset 7175.
   - `84b3adb7174dfd68`: `PAMM खाते में निवेश व्यापार रणनीति Combiner व्यावहारिक आवेदन` at offset 1520.
2. **Medicine:** drop keyword-stuffed spam (the spam rule of Medicine 2). In other documents remove only lines that are
   mostly hashtags or URLs. Comma lists of names or figures in news (`इस अवसर पर …, …, … आदि मौजूद रहे`) are content and stay.
   ```python
   TAG_OR_URL = re.compile(r"https?://\S+|www\.\S+|#[\w\u0900-\u097F]+")

   def cure_c5(text):
       if is_spam(text):
           return None
       keep = [line for line in text.split("\n")
               if not ((tags := TAG_OR_URL.findall(line)) and len(tags) >= 3
                       and sum(map(len, tags)) >= 0.5 * len(line.strip()))]
       return "\n".join(keep)
   ```
3. **Action:** drop the document (spam); otherwise fix (remove hashtag or link lines).
4. **Harm test:** pending (gold set). Pack check: cures 12 of 12 (11 dropped, 1 hashtag line removed); side effects 0 of 138; clean touched 0.
5. **Side effects:** spam in other topics (betting, adult, pharmacy) is not in the term list; the list must grow from
   reviewed pages. A real news article quoting many forex terms could be dropped; the two-distinct-terms rule made this 0 in the pack.
   A first version that also removed comma lists deleted news content in 4 clean documents and was rejected.

## Medicine 10: A6 Devanagari and Latin glued in one word

1. **Disease:** A6, 12 of 150 documents. Examples:
   - `39bd4d9f4387f55b`: `AnyOptionनवोन्मेशी` at offset 1542.
   - `0573df3b6b0cbc3c`: `रहूंगाbye` at offset 4489.
   - `c8d69b4928763080`: `अनुभवUniversity` at offset 32121.
2. **Medicine:** first cure fake dandas (Medicine 6, so `हैI` becomes `है।`), then insert a space where a Hindi
   letter touches a Latin word of 2+ letters.
   ```python
   A6_JOIN = re.compile(f"(?<=[{DEVA}])(?=[A-Za-z]{{2}})|(?<=[A-Za-z]{{2}})(?=[\u0904-\u0939\u0958-\u0961])")

   def cure_a6(text):
       return A6_JOIN.sub(" ", cure_g4(text))
   ```
   Examples: `AnyOptionनवोन्मेशी` → `AnyOption नवोन्मेशी`, `रहूंगाbye` → `रहूंगा bye`, `है।UNICAF` → `है। UNICAF`.
3. **Action:** fix the text.
4. **Harm test:** pending (gold set). Pack check: cures 12 of 12; side effects 18 of 138, all correct splits or the
   G4 changes above (`ReplyDeleteआदरणीय` → `ReplyDelete आदरणीय`); clean touched 0.
5. **Side effects:** a deliberate brand written across scripts would be split. All 12 A6 documents in the pack are
   dropped for other reasons (spam, machine translation), so the cure matters mostly for the documents kept.

## Medicine 11: G1 words split before a vowel sign

1. **Disease:** G1, 7 of 150 documents. Examples:
   - `a02caf8e245bbcf3`: `मे ंगुजार` at offset 3746 (`में गुजार`, space before the anusvara).
   - `8fed2d4464e4cc06`: `मे ं एक नि श् चि त` at offset 2789 (whole paragraph split at every sign).
   - `0f58a7ef798cc635`: `पाश् चात्य` at offset 1880.
2. **Medicine:** only in documents where the split is systematic (3 or more splits), join a vowel sign or anusvara to
   its letter, and join a halant to the next consonant.
   ```python
   G1_SPLIT = re.compile(f"([{CONS}\u093C])\\s+([{MATRA}\u0901-\u0903])")
   G1_ANUSVARA = re.compile(f"([{MATRA}])\\s+([\u0901\u0902])")
   G1_HALANT = re.compile(f"([{CONS}]\u094D)\\s+(?=[{CONS}])")

   def cure_g1(text):
       if len(G1_SPLIT.findall(text)) + len(G1_ANUSVARA.findall(text)) < 3:
           return text                                      # a single 'जगत् में' is correct Hindi
       text = G1_ANUSVARA.sub(lambda m: m.group(1) + m.group(2) + " ", text)   # 'मे ंगुजार' -> 'में गुजार'
       text = G1_SPLIT.sub(r"\1\2", text)
       text = re.sub("([\u0901\u0902]) (\\s)", r"\1\2", text)
       return G1_HALANT.sub(r"\1", text)
   ```
   Examples: `लाचारी मे ंगुजार देते` → `लाचारी में गुजार देते`; `प् रति शत … रू प मे ं` → `प्रति शत … रू प में`.
3. **Action:** fix the text.
4. **Harm test:** pending (gold set). Pack check: cures 2 of 7; side effects 0 of 143; clean touched 0.
5. **Side effects:** splits between two consonant syllables (`वि कि पी डि या`) cannot be joined without a word list,
   and `… [+]` truncation marks inside words (`ह… [+] ै`) are not handled; those 5 documents are dropped for other
   reasons. In a systematically split document a Sanskrit word ending in halant followed by another word would be merged.
