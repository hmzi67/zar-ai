# System Usability Scale: Urdu template

Draft translation of the standard SUS. **Have it back-translated and reviewed before the pilot.**

ہر بیان کے لیے ایک نمبر چنیں: 1 = بالکل متفق نہیں، 5 = مکمل متفق۔

(Choose one number for each statement: 1 = strongly disagree, 5 = strongly agree.)

| # | بیان | Statement | 1-5 |
|---|---|---|---|
| 1 | میرا خیال ہے کہ میں یہ نظام بار بار استعمال کرنا چاہوں گا/گی۔ | I think that I would like to use this system frequently. | |
| 2 | مجھے یہ نظام غیر ضروری طور پر پیچیدہ لگا۔ | I found the system unnecessarily complex. | |
| 3 | مجھے یہ نظام استعمال کرنا آسان لگا۔ | I thought the system was easy to use. | |
| 4 | میرا خیال ہے کہ اسے استعمال کرنے کے لیے مجھے کسی تکنیکی ماہر کی مدد چاہیے ہو گی۔ | I think that I would need the support of a technical person to be able to use this system. | |
| 5 | اس نظام کے مختلف حصے اچھی طرح آپس میں جڑے ہوئے لگے۔ | I found the various functions in this system were well integrated. | |
| 6 | مجھے اس نظام میں بہت زیادہ بے ربطی لگی۔ | I thought there was too much inconsistency in this system. | |
| 7 | میرا خیال ہے کہ زیادہ تر لوگ اسے جلدی استعمال کرنا سیکھ لیں گے۔ | I would imagine that most people would learn to use this system very quickly. | |
| 8 | مجھے یہ نظام استعمال کرنا بہت مشکل لگا۔ | I found the system very cumbersome to use. | |
| 9 | یہ نظام استعمال کرتے ہوئے مجھے بہت اعتماد محسوس ہوا۔ | I felt very confident using the system. | |
| 10 | اسے استعمال کرنے سے پہلے مجھے بہت کچھ سیکھنا پڑا۔ | I needed to learn a lot of things before I could get going with this system. | |

Record answers in `eval/pilot/pilot_log_template.csv` as `sus_q1` to `sus_q10`, then score them with `python -m zarai.eval.sus --input <file>`.
