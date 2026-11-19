# topic, evidence (exact substring of the KB), (q_ur, q_ro, q_en), (a_ur, a_ro, a_en)
F = [
    (
        "sowing_varieties",
        "آبپاش علاقوں میں گندم کا بہترین وقت کاشت یکم تا 20 نومبر ہے",
        (
            "نہری علاقے میں گندم کب کاشت کروں؟",
            "nehri ilaqe mein gandum kab kasht karoon?",
            "When should I sow wheat in an irrigated area?",
        ),
        (
            "آبپاش علاقوں میں گندم کا بہترین وقت کاشت یکم تا 20 نومبر ہے، تاہم 30 نومبر تک کاشت کی جا سکتی ہے۔",
            "Aabpash ilaqon mein gandum ka behtareen waqt-e-kasht 1 se 20 November hai, taham 30 November tak kasht ki ja sakti hai.",
            "In irrigated areas the best time to sow wheat is 1 to 20 November, though it can be sown until 30 November.",
        ),
    ),
    (
        "sowing_varieties",
        "بارانی علاقوں میں گندم کا سفارش کردہ وقت کاشت 15 اکتوبر تا 15 نومبر ہے",
        (
            "بارانی زمین میں گندم بونے کا صحیح ٹائم کیا ہے؟",
            "barani zameen mein gandum bonay ka sahi time kya hai?",
            "What is the right time to sow wheat on rainfed land?",
        ),
        (
            "بارانی علاقوں میں گندم کا سفارش کردہ وقت کاشت 15 اکتوبر تا 15 نومبر ہے۔",
            "Barani ilaqon mein gandum ka sifarish karda waqt-e-kasht 15 October se 15 November hai.",
            "In rainfed areas the recommended sowing time for wheat is 15 October to 15 November.",
        ),
    ),
    (
        "sowing_varieties",
        "یکم نومبر تا 20 نومبر | 40 تا 45 کلوگرام فی ایکڑ",
        (
            "ایک ایکڑ میں گندم کا کتنا بیج ڈالوں؟",
            "aik acre mein gandum ka kitna beej daloon?",
            "How much wheat seed should I use per acre?",
        ),
        (
            "یکم تا 20 نومبر کی کاشت کے لئے 40 تا 45 کلوگرام فی ایکڑ بیج رکھیں، بشرطیکہ اگاؤ کی شرح 85 فیصد سے زیادہ ہو۔ اگاؤ کم ہو تو شرح بیج مناسب حد تک بڑھا دیں۔",
            "1 se 20 November ki kasht ke liye 40 se 45 kilo fi acre beej rakhein, bashart-e-ke ugao ki sharah 85 fisad se zyada ho. Ugao kam ho to sharah-e-beej munasib had tak barha dein.",
            "For sowing between 1 and 20 November use 40 to 45 kg of seed per acre, provided germination is above 85 percent. If germination is lower, increase the seed rate accordingly.",
        ),
    ),
    (
        "sowing_varieties",
        "21 نومبر تا 10 دسمبر | 50 کلوگرام فی ایکڑ",
        (
            "دسمبر میں گندم لگا رہا ہوں، بیج کتنا رکھوں؟",
            "December mein gandum laga raha hoon, beej kitna rakhoon?",
            "I am sowing wheat in December. How much seed should I use?",
        ),
        (
            "21 نومبر سے 10 دسمبر تک کی کاشت کے لئے 50 کلوگرام فی ایکڑ بیج استعمال کریں۔ پچھیتی کاشت میں بیج کچھ بڑھانے سے شاخیں زیادہ بنتی ہیں اور جڑی بوٹیاں بھی کم اُگتی ہیں۔",
            "21 November se 10 December tak ki kasht ke liye 50 kilo fi acre beej istemal karein. Pichheti kasht mein beej kuch barhane se shaakhein zyada banti hain aur jari bootiyan bhi kam ugti hain.",
            "For sowing from 21 November to 10 December use 50 kg of seed per acre. A somewhat higher seed rate in late sowing gives more tillers and also suppresses weeds.",
        ),
    ),
    (
        "sowing_varieties",
        "شرح بیج 55 تا 60 کلوگرام فی ایکڑ رکھیں",
        (
            "کھڑی کپاس میں گندم کا چھٹہ دینا ہے، بیج کتنا لگے گا؟",
            "khari kapas mein gandum ka chhatta dena hai, beej kitna lagega?",
            "I want to broadcast wheat into standing cotton. How much seed is needed?",
        ),
        (
            "کھڑی کپاس میں گندم کی کاشت کے لئے شرح بیج 55 تا 60 کلوگرام فی ایکڑ رکھیں۔ بہترین وقت یکم تا 20 نومبر ہے اور کھیت کا جڑی بوٹیوں سے پاک ہونا ضروری ہے۔",
            "Khari kapas mein gandum ki kasht ke liye sharah-e-beej 55 se 60 kilo fi acre rakhein. Behtareen waqt 1 se 20 November hai aur khet ka jari bootiyon se paak hona zaroori hai.",
            "For wheat sown into standing cotton use 55 to 60 kg of seed per acre. The best time is 1 to 20 November, and the field must be free of weeds.",
        ),
    ),
    (
        "sowing_varieties",
        "پچھیتی کاشت میں بیج اُگنے میں زیادہ دن لیتا ہے",
        (
            "گندم لیٹ بونے سے کیا نقصان ہوتا ہے؟",
            "gandum late bonay se kya nuqsan hota hai?",
            "What is the harm in sowing wheat late?",
        ),
        (
            "پچھیتی کاشتہ گندم کی پیداوار روزانہ کی بنیاد پر کم ہوتی جاتی ہے۔ بیج دیر سے اُگتا ہے، پودا شگوفے کم بناتا ہے اور سٹے چھوٹے رہ جاتے ہیں، اور دانہ بنتے وقت گرمی بڑھے تو پیداوار اور کم ہو جاتی ہے۔",
            "Pichheti kasht gandum ki paidawar rozana ki buniyad par kam hoti jati hai. Beej der se ugta hai, pauda shagoofay kam banata hai aur sittay chhotay reh jatay hain, aur dana bantay waqt garmi barhe to paidawar aur kam ho jati hai.",
            "Yield of late-sown wheat falls day by day. Seed takes longer to emerge, plants form fewer tillers and ears stay small, and heat during grain formation cuts yield further.",
        ),
    ),
    (
        "sowing_varieties",
        "ڈیورم-2021 گندم کی الگ نوعیت کی قسم ہے",
        (
            "ڈیورم گندم کس کام آتی ہے؟",
            "durum gandum kis kaam aati hai?",
            "What is durum wheat used for?",
        ),
        (
            "ڈیورم-2021 گندم کی الگ نوعیت کی قسم ہے جو صنعتی استعمال یعنی پاستہ اور نوڈلز جیسی مصنوعات بنانے کے لئے موزوں ہے۔",
            "Durum-2021 gandum ki alag nauiyat ki qisam hai jo sanati istemal yani pasta aur noodles jaisi masnoo'at banane ke liye mozoon hai.",
            "Durum-2021 is a distinct type of wheat suited to industrial use, such as making pasta and noodles.",
        ),
    ),
    (
        "sowing_varieties",
        "اکبر-19، نواب-21 اور زنکول-16 غذائیت کے اعتبار سے زنک کی زیادہ مقدار",
        (
            "کون سی گندم میں زنک زیادہ ہوتا ہے؟",
            "kaun si gandum mein zinc zyada hota hai?",
            "Which wheat varieties are high in zinc?",
        ),
        (
            "اکبر-19، نواب-21 اور زنکول-16 غذائیت کے اعتبار سے زنک کی زیادہ مقدار والی اقسام ہیں۔",
            "Akbar-19, Nawab-21 aur Zincol-16 ghizaiyat ke aitbar se zinc ki zyada miqdar wali aqsam hain.",
            "Akbar-19, Nawab-21 and Zincol-16 are the varieties with higher zinc content.",
        ),
    ),
    (
        "sowing_varieties",
        "صادق-21 اور نواب-21 جنوبی پنجاب میں کاشت کے لئے زیادہ موزوں ہیں",
        (
            "جنوبی پنجاب کے لیے کون سی قسم اچھی ہے؟",
            "junoobi Punjab ke liye kaun si qisam achhi hai?",
            "Which variety suits South Punjab?",
        ),
        (
            "صادق-21 اور نواب-21 جنوبی پنجاب میں کاشت کے لئے زیادہ موزوں ہیں۔",
            "Sadiq-21 aur Nawab-21 junoobi Punjab mein kasht ke liye zyada mozoon hain.",
            "Sadiq-21 and Nawab-21 are the most suitable varieties for South Punjab.",
        ),
    ),
    (
        "sowing_varieties",
        "فیصل آباد-2008 اور پاکستان-13 کنگی سے متاثر ہوتی ہیں",
        (
            "کیا فیصل آباد 2008 ابھی بھی لگانی چاہیے؟",
            "kya Faisalabad 2008 abhi bhi lagani chahiye?",
            "Should I still grow Faisalabad-2008?",
        ),
        (
            "فیصل آباد-2008 اور پاکستان-13 کنگی سے متاثر ہوتی ہیں، اس لئے انہیں کم رقبہ پر کاشت کریں اور کنگی کے خلاف قوت مدافعت رکھنے والی نئی اقسام کو ترجیح دیں۔",
            "Faisalabad-2008 aur Pakistan-13 kungi se mutasir hoti hain, is liye inhein kam raqbay par kasht karein aur kungi ke khilaf quwwat-e-mudafiat rakhne wali nai aqsam ko tarjeeh dein.",
            "Faisalabad-2008 and Pakistan-13 are susceptible to rust, so grow them only on a small area and prefer newer rust-resistant varieties.",
        ),
    ),
    (
        "sowing_varieties",
        "عروج-22 و نشان-21 آبپاش و بارانی دونوں علاقہ جات کے لئے منظور شدہ",
        (
            "ایسی قسم بتائیں جو نہری اور بارانی دونوں جگہ چلے",
            "aisi qisam batayein jo nehri aur barani donon jagah chale",
            "Suggest a variety that works in both irrigated and rainfed areas.",
        ),
        (
            "عروج-22 اور نشان-21 آبپاش اور بارانی دونوں علاقوں کے لئے منظور شدہ اقسام ہیں۔",
            "Urooj-22 aur Nishan-21 aabpash aur barani donon ilaqon ke liye manzoor shuda aqsam hain.",
            "Urooj-22 and Nishan-21 are approved for both irrigated and rainfed areas.",
        ),
    ),
    (
        "sowing_varieties",
        "بھکر سٹار-19 کو 10 نومبر تا 10 دسمبر",
        (
            "بھکر سٹار کب کاشت کریں؟",
            "Bhakkar Star kab kasht karein?",
            "When should Bhakkar Star-19 be sown?",
        ),
        (
            "بھکر سٹار-19 کو 10 نومبر تا 10 دسمبر تک کاشت کریں۔",
            "Bhakkar Star-19 ko 10 November se 10 December tak kasht karein.",
            "Sow Bhakkar Star-19 between 10 November and 10 December.",
        ),
    ),
    (
        "sowing_varieties",
        "پنجاب سیڈ کارپوریشن اور رجسٹرڈ پرائیویٹ سیڈ کمپنیوں",
        (
            "گندم کا اچھا بیج کہاں سے ملے گا؟",
            "gandum ka achha beej kahan se milega?",
            "Where can I get good wheat seed?",
        ),
        (
            "ترقی دادہ اقسام کا بیج پنجاب سیڈ کارپوریشن اور رجسٹرڈ پرائیویٹ سیڈ کمپنیوں کے ڈپوؤں اور ڈیلروں سے دستیاب ہے۔ خریدتے وقت فیڈرل سیڈ سرٹیفیکیشن کا ٹیگ ضرور دیکھیں۔",
            "Taraqqi dada aqsam ka beej Punjab Seed Corporation aur registered private seed companiyon ke depo'on aur dealeron se dastiyab hai. Kharidte waqt Federal Seed Certification ka tag zaroor dekhein.",
            "Seed of improved varieties is sold at Punjab Seed Corporation and registered private seed company depots and dealers. Check the Federal Seed Certification tag when buying.",
        ),
    ),
    (
        "disease",
        "تھائیوفینیٹ میتھائل بحساب دو تا اڑھائی گرام",
        (
            "بوائی سے پہلے بیج کو کون سی دوائی لگاؤں؟",
            "bowai se pehle beej ko kaun si dawai lagaoon?",
            "Which treatment should I put on the seed before sowing?",
        ),
        (
            "کانگیاری، کرنال بنٹ، بلاسٹ اور اکھیڑا سے بچاؤ کے لئے بیج کو بوائی سے پہلے تھائیوفینیٹ میتھائل بحساب دو تا اڑھائی گرام، یا امیڈاکلوپرڈ + ٹیبوکونازول بحساب 2 ملی لیٹر فی کلوگرام بیج لگائیں۔ لیبل کی ہدایات بھی پڑھیں۔",
            "Kangiyari, Karnal bunt, blast aur ukhera se bachao ke liye beej ko bowai se pehle thiophanate methyl bahisab do se dhai gram, ya imidacloprid + tebuconazole bahisab 2 ml fi kilo beej lagayein. Label ki hidayaat bhi parhein.",
            "To protect against smut, Karnal bunt, blast and foot rot, treat seed before sowing with thiophanate methyl at 2 to 2.5 g, or imidacloprid + tebuconazole at 2 ml, per kg of seed. Also read the label.",
        ),
    ),
    (
        "disease",
        "پلاسٹک کی ایک بوری میں وزن شدہ بیج",
        (
            "ڈرم نہیں ہے، بیج کو زہر کیسے لگاؤں؟",
            "drum nahin hai, beej ko zehar kaise lagaoon?",
            "I have no drum. How do I treat seed?",
        ),
        (
            "پلاسٹک کی بوری میں وزن شدہ بیج اور سفارش کردہ دوا ڈال کر منہ باندھ دیں اور دونوں طرف سے پکڑ کر اچھی طرح ہلائیں۔ بوری تقریباً آدھی بھریں تاکہ ہر دانے کو دوا لگ جائے۔",
            "Plastic ki bori mein wazan shuda beej aur sifarish karda dawa daal kar munh bandh dein aur donon taraf se pakar kar achhi tarah hilayein. Bori taqreeban aadhi bharein taake har danay ko dawa lag jaye.",
            "Put the weighed seed and the recommended product in a plastic sack, tie it shut and shake it well holding both ends. Fill the sack only about half way so every grain gets coated.",
        ),
    ),
    (
        "sowing_varieties",
        "بیج اڑھائی انچ سے زیادہ گہرائی تک نہیں جانا چاہیے",
        (
            "ڈرل سے بیج کتنا گہرا ڈالوں؟",
            "drill se beej kitna gehra daloon?",
            "How deep should the drill place the seed?",
        ),
        (
            "ڈرل سے کاشت میں بیج اڑھائی انچ سے زیادہ گہرا نہیں جانا چاہیے، ورنہ بیج دیر سے اُگتا ہے اور شگوفے کم بنتے ہیں۔",
            "Drill se kasht mein beej dhai inch se zyada gehra nahin jana chahiye, warna beej der se ugta hai aur shagoofay kam bantay hain.",
            "With a drill, seed should not go deeper than 2.5 inches, otherwise it emerges late and forms fewer tillers.",
        ),
    ),
    (
        "fertiliser",
        "نائٹروجنی کھاد مثلاً یوریا وغیرہ ڈرل کے ذریعے استعمال نہ کریں",
        (
            "کیا یوریا ڈرل میں ڈال سکتے ہیں؟",
            "kya urea drill mein daal saktay hain?",
            "Can I put urea through the drill?",
        ),
        (
            "نہیں۔ یوریا جیسی نائٹروجنی کھاد ڈرل کے ذریعے استعمال نہ کریں، اور بیج اور کھاد کو کبھی مکس نہ کریں بلکہ الگ الگ بکس میں ڈالیں۔",
            "Nahin. Urea jaisi nitrogeni khaad drill ke zariye istemal na karein, aur beej aur khaad ko kabhi mix na karein balkay alag alag box mein dalein.",
            "No. Do not apply nitrogen fertilizer such as urea through the drill, and never mix seed and fertilizer, use separate boxes.",
        ),
    ),
    (
        "fertiliser",
        "اپنے ضلع میں موجود لیبارٹری سے زمین کا تجزیہ کروا کر",
        (
            "گندم کو کھاد کتنی ڈالوں؟",
            "gandum ko khaad kitni daloon?",
            "How much fertilizer should I give my wheat?",
        ),
        (
            "کھاد اپنے ضلع کی لیبارٹری سے زمین کا تجزیہ کروا کر ڈالیں، کیونکہ مقدار زمین کی زرخیزی، کلراٹھے پن، پانی اور پچھلی فصل پر منحصر ہے۔ «کھاد حساب» موبائل ایپ سے بھی حساب لگا سکتے ہیں۔",
            'Khaad apne zila ki laboratory se zameen ka tajziya karwa kar dalein, kyunke miqdar zameen ki zarkhezi, kallarathay pan, pani aur pichli fasal par munhasir hai. "Khaad Hisab" mobile app se bhi hisab laga saktay hain.',
            'Base fertilizer on a soil test from your district laboratory, because the amount depends on soil fertility, salinity, water and the previous crop. You can also use the "Khaad Hisab" mobile app.',
        ),
    ),
    (
        "fertiliser",
        "ڈیڑھ بوری ڈی اے پی + پونے دو بوری یوریا + ایک بوری ایس او پی/ایم او پی",
        (
            "درمیانی زمین ہے، نہری علاقہ، کون سی کھاد کتنی ڈالوں؟",
            "darmiyani zameen hai, nehri ilaqa, kaun si khaad kitni daloon?",
            "My irrigated land is of average fertility. Which fertilizer and how much?",
        ),
        (
            "آبپاش علاقے میں اوسط زرخیز زمین کے لئے ایک سفارش ڈیڑھ بوری ڈی اے پی + پونے دو بوری یوریا + ایک بوری ایس او پی یا ایم او پی فی ایکڑ ہے۔ بہتر ہے پہلے زمین کا تجزیہ کروا لیں۔",
            "Aabpash ilaqay mein ausat zarkhez zameen ke liye aik sifarish dairh bori DAP + paunay do bori urea + aik bori SOP ya MOP fi acre hai. Behtar hai pehle zameen ka tajziya karwa lein.",
            "For average-fertility irrigated land, one recommendation is 1.5 bags DAP + 1.75 bags urea + 1 bag SOP or MOP per acre. Get a soil test first if you can.",
        ),
    ),
    (
        "fertiliser",
        "دو بوری ڈی اے پی + دو بوری یوریا + ایک بوری ایس او پی/ایم او پی",
        (
            "کمزور زمین کے لیے کھاد کی سفارش کیا ہے؟",
            "kamzor zameen ke liye khaad ki sifarish kya hai?",
            "What is the fertilizer recommendation for weak soil?",
        ),
        (
            "آبپاش علاقے کی کمزور زمین کے لئے ایک سفارش دو بوری ڈی اے پی + دو بوری یوریا + ایک بوری ایس او پی یا ایم او پی فی ایکڑ ہے۔",
            "Aabpash ilaqay ki kamzor zameen ke liye aik sifarish do bori DAP + do bori urea + aik bori SOP ya MOP fi acre hai.",
            "For weak irrigated soil, one recommendation is 2 bags DAP + 2 bags urea + 1 bag SOP or MOP per acre.",
        ),
    ),
    (
        "fertiliser",
        "نائٹروجنی کھاد 2 یا 3 اقساط میں استعمال کریں",
        (
            "یوریا ایک ہی دفعہ ڈال دوں یا قسطوں میں؟",
            "urea aik hi dafa daal doon ya qiston mein?",
            "Should I apply urea all at once or in splits?",
        ),
        (
            "فاسفورس اور پوٹاش کی ساری کھاد اور نائٹروجن کی پہلی قسط بوائی کے وقت ڈالیں۔ نائٹروجنی کھاد 2 یا 3 قسطوں میں دیں، اور ریتلی زمین میں چار یا زیادہ قسطوں میں۔",
            "Phosphorus aur potash ki saari khaad aur nitrogen ki pehli qist bowai ke waqt dalein. Nitrogeni khaad 2 ya 3 qiston mein dein, aur reteeli zameen mein chaar ya zyada qiston mein.",
            "Apply all the phosphorus and potash and the first nitrogen split at sowing. Give nitrogen in 2 or 3 splits, or 4 or more on sandy soil.",
        ),
    ),
    (
        "fertiliser",
        "بارانی علاقوں میں کھاد کی ساری مقدار بوقت بوائی استعمال کریں",
        (
            "بارانی گندم میں کھاد کب ڈالیں؟",
            "barani gandum mein khaad kab dalein?",
            "When should fertilizer go on rainfed wheat?",
        ),
        (
            "بارانی علاقوں میں کھاد کی ساری مقدار بوائی کے وقت ہی ڈالیں۔",
            "Barani ilaqon mein khaad ki saari miqdar bowai ke waqt hi dalein.",
            "In rainfed areas apply all the fertilizer at sowing.",
        ),
    ),
    (
        "fertiliser",
        "زنک سلفیٹ (ZnSO4) 33 فیصد بحساب 5 کلوگرام",
        (
            "گندم میں زنک کتنا ڈالیں؟",
            "gandum mein zinc kitna dalein?",
            "How much zinc should I apply to wheat?",
        ),
        (
            "بوائی کے وقت زنک سلفیٹ 33 فیصد بحساب 5 کلوگرام اور بورک ایسڈ 17 فیصد بحساب 2.50 کلوگرام فی ایکڑ ڈالیں۔ اگر پچھلی فصل میں زنک سلفیٹ ڈالا تھا تو گندم میں نہ ڈالیں۔",
            "Bowai ke waqt zinc sulphate 33 fisad bahisab 5 kilo aur boric acid 17 fisad bahisab 2.50 kilo fi acre dalein. Agar pichli fasal mein zinc sulphate dala tha to gandum mein na dalein.",
            "At sowing apply zinc sulphate (33%) at 5 kg and boric acid (17%) at 2.5 kg per acre. Skip the zinc if it was applied to the previous crop.",
        ),
    ),
    (
        "fertiliser",
        "کے نام سے ایک موبائل ایپ تیار کی ہے",
        (
            "کھاد کا حساب لگانے کی کوئی ایپ ہے؟",
            "khaad ka hisab lagane ki koi app hai?",
            "Is there an app to calculate fertilizer?",
        ),
        (
            "جی ہاں۔ ادارہ زرخیزی زمین پنجاب لاہور کی «کھاد حساب» ایپ مفت ہے۔ گوگل پلے سٹور میں SFRI لکھ کر ڈاؤن لوڈ کریں۔ یہ زمین کے تجزیہ کے ساتھ یا اس کے بغیر کھاد کی مقدار بتاتی ہے۔",
            'Ji haan. Idara Zarkhezi Zameen Punjab Lahore ki "Khaad Hisab" app muft hai. Google Play Store mein SFRI likh kar download karein. Yeh zameen ke tajziye ke saath ya us ke baghair khaad ki miqdar batati hai.',
            'Yes. The Soil Fertility Research Institute\'s free "Khaad Hisab" app is on Google Play (search SFRI). It gives fertilizer amounts with or without a soil test report.',
        ),
    ),
    (
        "fertiliser",
        "گوبر کی گلی سڑی کھاد بحساب 8 تا 10 ٹن",
        (
            "گوبر کی کھاد کتنی ڈالیں؟",
            "gobar ki khaad kitni dalein?",
            "How much farmyard manure should I apply?",
        ),
        (
            "گوبر کی گلی سڑی کھاد 8 تا 10 ٹن یعنی تین سے چار ٹرالی فی ایکڑ ڈالیں۔ کچی کھاد نہ ڈالیں کیونکہ اس سے دیمک بڑھتی ہے۔",
            "Gobar ki gali sari khaad 8 se 10 ton yani teen se chaar trolley fi acre dalein. Kachi khaad na dalein kyunke is se deemak barhti hai.",
            "Apply well-rotted manure at 8 to 10 tonnes, about three to four trolleys, per acre. Do not use raw manure, as it encourages termites.",
        ),
    ),
    (
        "irrigation",
        "پہلا پانی: شگوفے نکلتے وقت یعنی بوائی کے 20 تا 25 دن بعد",
        (
            "کپاس کے بعد والی گندم کو پہلا پانی کب لگاؤں؟",
            "kapas ke baad wali gandum ko pehla pani kab lagaoon?",
            "When should wheat after cotton get its first irrigation?",
        ),
        (
            "کپاس، مکئی یا کماد کے بعد کاشتہ گندم کو پہلا پانی بوائی کے 20 تا 25 دن بعد شگوفے نکلتے وقت دیں۔ دیر کرنے سے شگوفے اور سٹے کم بنتے ہیں۔",
            "Kapas, makai ya kamad ke baad kasht gandum ko pehla pani bowai ke 20 se 25 din baad shagoofay nikaltay waqt dein. Der karne se shagoofay aur sittay kam bantay hain.",
            "Wheat after cotton, maize or sugarcane should get its first irrigation 20 to 25 days after sowing, when tillers emerge. Delay means fewer tillers and ears.",
        ),
    ),
    (
        "irrigation",
        "پہلا پانی: بوائی کے 35 تا 45 دن بعد",
        (
            "دھان کے بعد گندم کو پہلا پانی کب دیں؟",
            "dhaan ke baad gandum ko pehla pani kab dein?",
            "When does wheat after rice need its first irrigation?",
        ),
        (
            "دھان کے بعد کاشتہ گندم کو پہلا پانی بوائی کے 35 تا 45 دن بعد دیں، کیونکہ دھان کی زمین میں نمی زیادہ ہوتی ہے۔",
            "Dhaan ke baad kasht gandum ko pehla pani bowai ke 35 se 45 din baad dein, kyunke dhaan ki zameen mein nami zyada hoti hai.",
            "Wheat after rice should get its first irrigation 35 to 45 days after sowing, because rice fields hold more moisture.",
        ),
    ),
    (
        "irrigation",
        "دوسرا پانی: بوائی کے 80 تا 90 دن بعد (گوبھ کے وقت)",
        (
            "گندم کو دوسرا پانی کب دینا ہے؟",
            "gandum ko doosra pani kab dena hai?",
            "When is the second irrigation due?",
        ),
        (
            "دوسرا پانی بوائی کے 80 تا 90 دن بعد گوبھ کے وقت دیں۔ اس وقت پانی نہ ملے تو سٹے چھوٹے رہتے ہیں اور دانے کم بنتے ہیں۔",
            "Doosra pani bowai ke 80 se 90 din baad gobh ke waqt dein. Is waqt pani na milay to sittay chhotay rehtay hain aur danay kam bantay hain.",
            "Give the second irrigation 80 to 90 days after sowing, at booting. Without water then, ears stay small with fewer grains.",
        ),
    ),
    (
        "irrigation",
        "دانہ بننے کی ابتداء یعنی بوائی کے 125 تا 130 دن بعد (دودھیا حالت)",
        (
            "دودھیا حالت میں پانی ضروری ہے؟",
            "doodhiya halat mein pani zaroori hai?",
            "Is irrigation needed at the milky stage?",
        ),
        (
            "جی ہاں۔ تیسرا پانی دانہ بننے کی ابتدا یعنی بوائی کے 125 تا 130 دن بعد دودھیا حالت پر دیں، ورنہ دانہ چھوٹا اور وزن کم رہ جاتا ہے۔",
            "Ji haan. Teesra pani dana banne ki ibtida yani bowai ke 125 se 130 din baad doodhiya halat par dein, warna dana chhota aur wazan kam reh jata hai.",
            "Yes. Give the third irrigation at the start of grain formation, 125 to 130 days after sowing (milky stage), or grains stay small and light.",
        ),
    ),
    (
        "irrigation",
        "25 مارچ تک آخری آبپاشی مکمل کر لیں",
        (
            "گندم کو آخری پانی کب تک لگانا ہے؟",
            "gandum ko aakhri pani kab tak lagana hai?",
            "By when should the last irrigation be done?",
        ),
        (
            "آخری آبپاشی 25 مارچ تک مکمل کر لیں، تاہم موسم اور فصل کی ضرورت کے مطابق اس کے بعد بھی پانی لگایا جا سکتا ہے۔",
            "Aakhri aabpashi 25 March tak mukammal kar lein, taham mausam aur fasal ki zaroorat ke mutabiq is ke baad bhi pani lagaya ja sakta hai.",
            "Finish the last irrigation by 25 March, though water can still be given later if weather and the crop require it.",
        ),
    ),
    (
        "irrigation",
        (
            "تقریباً 6 مرتبہ آبپاشی کی ضرورت ہوتی ہے",
            "چھٹا پانی: دانے بھرتے وقت (بوائی کے 130 تا 135 دن بعد)",
        ),
        (
            "تھل کے علاقے میں گندم کو کتنے پانی چاہئیں؟",
            "Thal ke ilaqay mein gandum ko kitne pani chahiyein?",
            "How many irrigations does wheat need in the Thal area?",
        ),
        (
            "تھل اور دوسرے گرم ریتلے علاقوں میں گندم کو عموماً تقریباً 6 مرتبہ آبپاشی کی ضرورت ہوتی ہے: بوائی کے 20 تا 25، 40 تا 45، 70 تا 75، 90 تا 95، 110 تا 115 اور 130 تا 135 دن بعد۔",
            "Thal aur doosray garam reteelay ilaqon mein gandum ko umooman taqreeban 6 martaba aabpashi ki zaroorat hoti hai: bowai ke 20-25, 40-45, 70-75, 90-95, 110-115 aur 130-135 din baad.",
            "In Thal and other hot sandy areas wheat usually needs about 6 irrigations: 20-25, 40-45, 70-75, 90-95, 110-115 and 130-135 days after sowing.",
        ),
    ),
    (
        "irrigation",
        "اگر موسم متواتر گرم اور خشک رہے تو پہلے اور دوسرے",
        (
            "موسم بہت خشک ہے، کیا اضافی پانی دوں؟",
            "mausam bohat khushk hai, kya izafi pani doon?",
            "The weather is very dry. Should I give extra water?",
        ),
        (
            "اگر موسم متواتر گرم اور خشک رہے تو پہلے اور دوسرے، اور دوسرے اور تیسرے پانی کے درمیان ضرورت کے مطابق ایک زائد پانی لگائیں۔",
            "Agar mausam mutawatir garam aur khushk rahe to pehle aur doosray, aur doosray aur teesray pani ke darmiyan zaroorat ke mutabiq aik zaid pani lagayein.",
            "If it stays hot and dry, give one extra irrigation as needed between the first and second, and between the second and third.",
        ),
    ),
    (
        "irrigation",
        "لیزر لینڈ لیولر استعمال کیا جائے",
        (
            "کھیت ہموار کرنے کا کیا فائدہ ہے؟",
            "khet hamwar karne ka kya faida hai?",
            "What is the benefit of levelling the field?",
        ),
        (
            "ہموار کھیت میں پانی، کھاد اور اگاؤ یکساں ہوتے ہیں اور پہلا پانی صحیح مقدار میں لگتا ہے۔ اس کے لئے بہتر ہے لیزر لینڈ لیولر استعمال کریں۔",
            "Hamwar khet mein pani, khaad aur ugao yaksan hotay hain aur pehla pani sahi miqdar mein lagta hai. Is ke liye behtar hai laser land leveller istemal karein.",
            "A level field gets even water, fertilizer and emergence, and the first irrigation goes on in the right amount. A laser land leveller is best for this.",
        ),
    ),
    (
        "weed",
        "پہلے پانی کے بعد چوڑے پتے والی جڑی بوٹیوں",
        (
            "جڑی بوٹی مار سپرے کب کریں؟",
            "jari booti maar spray kab karein?",
            "When should I spray herbicide?",
        ),
        (
            "چوڑے پتے والی جڑی بوٹیوں کے لئے پہلے پانی کے بعد اور نوکیلے پتے والی کے لئے دوسرے پانی کے بعد سپرے کریں۔ دونوں کے لئے زہریں بھی ہیں جو تقریباً 40 تا 45 دن کی فصل پر لگتی ہیں۔ زہر کا انتخاب محکمہ زراعت کے مشورے سے کریں۔",
            "Chauray pattay wali jari bootiyon ke liye pehle pani ke baad aur nokeelay pattay wali ke liye doosray pani ke baad spray karein. Donon ke liye zehrein bhi hain jo taqreeban 40 se 45 din ki fasal par lagti hain. Zehar ka intikhab mehkma zaraat ke mashware se karein.",
            "Spray broadleaf weeds after the first irrigation and grassy weeds after the second. Products for both exist, applied at about 40 to 45 days. Choose the product with the agriculture department's advice.",
        ),
    ),
    (
        "weed",
        "پانی کی مقدار 100 تا 120 لیٹر فی ایکڑ رکھیں",
        (
            "سپرے میں ایک ایکڑ پر کتنا پانی لگتا ہے؟",
            "spray mein aik acre par kitna pani lagta hai?",
            "How much water per acre for a herbicide spray?",
        ),
        (
            "مشین کی کیلیبریشن کر کے سپرے کریں اور پانی کی مقدار 100 تا 120 لیٹر فی ایکڑ رکھیں۔ کہیں دوہرا سپرے نہ کریں اور کوئی جگہ خالی نہ چھوڑیں۔",
            "Machine ki calibration kar ke spray karein aur pani ki miqdar 100 se 120 litre fi acre rakhein. Kahin dohra spray na karein aur koi jagah khaali na chhorein.",
            "Calibrate the sprayer and use 100 to 120 litres of water per acre. Do not double-spray or leave gaps.",
        ),
    ),
    (
        "weed",
        "تیز ہوا، دھند یا بارش میں سپرے نہ کریں",
        ("دھند میں سپرے کر سکتے ہیں؟", "dhund mein spray kar saktay hain?", "Can I spray in fog?"),
        (
            "نہیں۔ تیز ہوا، دھند یا بارش میں سپرے نہ کریں۔ فلیٹ فین یا ٹی جیٹ نوزل استعمال کریں اور ترجیحاً دھوپ میں جب پتوں سے شبنم ختم ہو جائے تب سپرے کریں۔",
            "Nahin. Tez hawa, dhund ya barish mein spray na karein. Flat fan ya T-jet nozzle istemal karein aur tarjeehan dhoop mein jab patton se shabnam khatam ho jaye tab spray karein.",
            "No. Do not spray in strong wind, fog or rain. Use a flat fan or T-jet nozzle and spray preferably in sunshine once the dew has dried.",
        ),
    ),
    (
        "weed",
        "20 تا 25 فیصد تک اضافہ ہو سکتا ہے",
        (
            "جڑی بوٹیاں ختم کرنے سے کتنا فرق پڑتا ہے؟",
            "jari bootiyan khatam karne se kitna farq parta hai?",
            "How much difference does weed control make?",
        ),
        (
            "محکمہ زراعت کے مطابق جڑی بوٹیوں پر مؤثر قابو پانے سے موجودہ وسائل میں ہی گندم کی پیداوار 20 تا 25 فیصد تک بڑھ سکتی ہے۔",
            "Mehkma zaraat ke mutabiq jari bootiyon par moassar qaboo panay se maujooda wasail mein hi gandum ki paidawar 20 se 25 fisad tak barh sakti hai.",
            "According to the agriculture department, good weed control can raise wheat yield by 20 to 25 percent with existing resources.",
        ),
    ),
    (
        "pest",
        "وسط فروری تا مارچ میں اس کا حملہ زیادہ ہوتا ہے",
        (
            "گندم پر سبز چھوٹے کیڑے ہیں، یہ کیا ہیں؟",
            "gandum par sabz chhotay keeray hain, yeh kya hain?",
            "There are small green insects on my wheat. What are they?",
        ),
        (
            "یہ غالباً سست تیلہ ہے: سبزی مائل چھوٹا کیڑا جس کا جسم ناشپاتی جیسا ہوتا ہے۔ یہ پتوں اور سٹوں کا رس چوستا ہے اور اس کا حملہ وسط فروری تا مارچ زیادہ ہوتا ہے۔ یقین کے لئے محکمہ زراعت کو دکھائیں۔",
            "Yeh ghaliban sust tela hai: sabzi mail chhota keera jis ka jism nashpati jaisa hota hai. Yeh patton aur sitton ka ras choosta hai aur is ka hamla wast February se March zyada hota hai. Yaqeen ke liye mehkma zaraat ko dikhayein.",
            "These are probably aphids: small greenish, pear-shaped insects that suck sap from leaves and ears, mostly from mid-February to March. Show a sample to the agriculture department to confirm.",
        ),
    ),
    (
        "pest",
        "100 فٹ کے فاصلے پر سرسوں کی 2 لائنیں کاشت کریں",
        (
            "تیلے سے بچنے کے لئے کیا کروں؟",
            "telay se bachne ke liye kya karoon?",
            "How can I protect wheat from aphids?",
        ),
        (
            "گندم کے کھیت میں 100 فٹ کے فاصلے پر سرسوں یا کینولا کی 2 لائنیں کاشت کریں۔ مفید کیڑے پہلے سرسوں پر آتے ہیں اور پھر گندم کا تیلہ کھا جاتے ہیں۔ بروقت کاشت اور مناسب کھاد بھی حملہ کم کرتی ہے۔",
            "Gandum ke khet mein 100 foot ke faslay par sarson ya canola ki 2 lainein kasht karein. Mufeed keeray pehle sarson par aatay hain aur phir gandum ka tela kha jatay hain. Barwaqt kasht aur munasib khaad bhi hamla kam karti hai.",
            "Sow 2 rows of mustard or canola every 100 feet in the wheat field. Beneficial insects build up on the mustard first and then eat the aphids on wheat. Timely sowing and balanced fertilizer also reduce attacks.",
        ),
    ),
    (
        "pest",
        "شدید حملے کی صورت میں پریشر کے ساتھ پانی کا سپرے کریں",
        (
            "تیلہ بہت زیادہ ہو گیا ہے، کیا کروں؟",
            "tela bohat zyada ho gaya hai, kya karoon?",
            "Aphids have become very heavy. What should I do?",
        ),
        (
            "شدید حملے کی صورت میں پریشر کے ساتھ پانی کا سپرے کریں۔ لیڈی برڈ بیٹل جیسے مفید کیڑے عموماً تیلے کو بڑھنے نہیں دیتے۔ مزید مشورے کے لئے محکمہ زراعت سے رابطہ کریں۔",
            "Shadeed hamle ki surat mein pressure ke saath pani ka spray karein. Lady bird beetle jaise mufeed keeray umooman telay ko barhne nahin detay. Mazeed mashware ke liye mehkma zaraat se rabta karein.",
            "For a heavy attack, spray water under pressure. Beneficial insects such as ladybird beetles usually keep aphids in check. Contact the agriculture department for further advice.",
        ),
    ),
    (
        "pest",
        "گوبر کی کچی کھاد استعمال نہ کریں",
        (
            "پودے پیلے ہو کر مر رہے ہیں، جڑوں میں دیمک ہے",
            "paude peelay ho kar mar rahe hain, jarron mein deemak hai",
            "Plants are yellowing and dying and there are termites at the roots.",
        ),
        (
            "دیمک جڑوں پر حملہ کرتی ہے اور حملہ ٹکڑیوں میں ہوتا ہے۔ رائونی کے وقت یا حملے کے بعد آبپاشی کے ساتھ سفارش کردہ زہر محکمہ زراعت کے مشورے سے استعمال کریں، اور گوبر کی کچی کھاد استعمال نہ کریں۔",
            "Deemak jarron par hamla karti hai aur hamla tukriyon mein hota hai. Rauni ke waqt ya hamle ke baad aabpashi ke saath sifarish karda zehar mehkma zaraat ke mashware se istemal karein, aur gobar ki kachi khaad istemal na karein.",
            "Termites attack the roots, usually in patches. Use the recommended product at pre-sowing irrigation or with irrigation after an attack, on the agriculture department's advice, and avoid raw manure.",
        ),
    ),
    (
        "pest",
        "دھان کی کٹائی کے بعد مڈھ تلف کر دیں",
        (
            "گندم کی درمیانی کونپل سوکھ رہی ہے اور سٹہ سفید ہے",
            "gandum ki darmiyani konpal sookh rahi hai aur sitta safaid hai",
            "The central shoot is drying and the ear is white.",
        ),
        (
            "چاول کے علاقے میں یہ گلابی سنڈی ہو سکتی ہے جو درمیانی کونپل کاٹ دیتی ہے۔ متاثرہ شاخیں اکھاڑ کر تلف کریں اور دھان کی کٹائی کے بعد مڈھ تلف کر دیں۔ یقین کے لئے محکمہ زراعت کو دکھائیں۔",
            "Chawal ke ilaqay mein yeh gulabi sundi ho sakti hai jo darmiyani konpal kaat deti hai. Mutasira shaakhein ukhaar kar talaf karein aur dhaan ki katai ke baad mudh talaf kar dein. Yaqeen ke liye mehkma zaraat ko dikhayein.",
            "In rice areas this may be pink borer, which cuts the central shoot. Pull out and destroy affected tillers, and destroy rice stubble after harvest. Show it to the agriculture department to confirm.",
        ),
    ),
    (
        "pest",
        "حملہ شدہ کھیت کے گرد کھالی بنا کر",
        (
            "لشکری سنڈی ایک کھیت سے دوسرے میں جا رہی ہے",
            "lashkari sundi aik khet se doosray mein ja rahi hai",
            "Armyworms are moving from one field to the next.",
        ),
        (
            "حملہ شدہ کھیت کے گرد کھالی بنا کر سفارش کردہ زہر کا دھوڑا کریں تاکہ سنڈیاں دوسرے کھیتوں میں نہ جائیں۔ زہر کا انتخاب محکمہ زراعت کے مشورے سے کریں۔",
            "Hamla shuda khet ke gird khaali bana kar sifarish karda zehar ka dhoora karein taake sundiyan doosray kheton mein na jayein. Zehar ka intikhab mehkma zaraat ke mashware se karein.",
            "Dig a trench around the attacked field and dust it with the recommended product so the caterpillars cannot spread. Choose the product with the agriculture department's advice.",
        ),
    ),
    (
        "disease",
        "زرد کنگی میں زرد رنگ کے چھوٹے چھوٹے دھبے متوازی قطاروں میں",
        (
            "زرد کنگی اور بھوری کنگی میں فرق کیسے کروں؟",
            "zard kungi aur bhoori kungi mein farq kaise karoon?",
            "How do I tell yellow rust from brown rust?",
        ),
        (
            "زرد کنگی میں زرد رنگ کے چھوٹے دھبے متوازی قطاروں میں ہوتے ہیں۔ بھوری کنگی میں بھورے دھبے بے ترتیب بکھرے ہوتے ہیں اور پتوں پر بھورا پاؤڈر نظر آتا ہے۔",
            "Zard kungi mein zard rang ke chhotay dhabbe mutawazi qataron mein hotay hain. Bhoori kungi mein bhooray dhabbe be-tarteeb bikhray hotay hain aur patton par bhoora powder nazar aata hai.",
            "Yellow rust has small yellow pustules in parallel stripes. Brown rust has scattered brown pustules that leave brown powder on the leaves.",
        ),
    ),
    (
        "disease",
        "زیادہ رقبہ پر ایک قسم کی بجائے تین چار اقسام کاشت کریں",
        ("کنگی سے بچاؤ کیسے کریں؟", "kungi se bachao kaise karein?", "How can I prevent rust?"),
        (
            "کنگی کے خلاف مدافعت رکھنے والی سفارش کردہ اقسام لگائیں، بڑے رقبے پر ایک کی بجائے تین چار اقسام لگائیں اور کاشت نومبر میں کریں۔ غیر ضروری پانی، زیادہ بیج اور زیادہ نائٹروجن سے بچیں اور پوٹاش کی کھاد استعمال کریں۔",
            "Kungi ke khilaf mudafiat rakhne wali sifarish karda aqsam lagayein, baray raqbay par aik ki bajaye teen chaar aqsam lagayein aur kasht November mein karein. Ghair zaroori pani, zyada beej aur zyada nitrogen se bachein aur potash ki khaad istemal karein.",
            "Grow recommended rust-resistant varieties, use three or four varieties instead of one on large areas, and sow in November. Avoid unnecessary irrigation, high seed rates and excess nitrogen, and apply potash.",
        ),
    ),
    (
        "disease",
        "یاد رکھیں صرف پتوں کا پیلا رنگ کنگی کی علامت نہیں ہے",
        (
            "پتے پیلے ہیں، کیا یہ کنگی ہے؟",
            "pattay peelay hain, kya yeh kungi hai?",
            "The leaves are yellow. Is it rust?",
        ),
        (
            "ضروری نہیں۔ صرف پتوں کا پیلا ہونا کنگی کی علامت نہیں ہے۔ کنگی میں پتوں پر زرد یا بھورے دھبے اور پاؤڈر ہوتا ہے۔ یقین نہ ہو تو محکمہ زراعت کو دکھائیں۔",
            "Zaroori nahin. Sirf patton ka peela hona kungi ki alamat nahin hai. Kungi mein patton par zard ya bhooray dhabbe aur powder hota hai. Yaqeen na ho to mehkma zaraat ko dikhayein.",
            "Not necessarily. Yellow leaves alone are not a sign of rust. Rust shows yellow or brown pustules with powder. If unsure, show it to the agriculture department.",
        ),
    ),
    (
        "disease",
        "بیج کا کچھ حصہ پھپھوندی سے متاثر ہو کر سیاہ پاؤڈر بن جاتا ہے",
        ("کرنال بنٹ کیا ہوتا ہے؟", "Karnal bunt kya hota hai?", "What is Karnal bunt?"),
        (
            "کرنال بنٹ ایک پھپھوندی کی بیماری ہے جس میں چند سٹوں کے چند دانوں کا کچھ حصہ سیاہ پاؤڈر بن جاتا ہے۔ مدافعتی اقسام لگائیں، بیج کو سفارش کردہ دوا لگائیں، متاثرہ فصل کا بیج ہرگز استعمال نہ کریں اور فصلوں کا ہیر پھیر کریں۔",
            "Karnal bunt aik phaphoondi ki bimari hai jis mein chand sitton ke chand danon ka kuch hissa siyah powder ban jata hai. Mudafiati aqsam lagayein, beej ko sifarish karda dawa lagayein, mutasira fasal ka beej hargiz istemal na karein aur faslon ka hair pher karein.",
            "Karnal bunt is a fungal disease in which part of some grains in some ears turns to black powder. Grow resistant varieties, treat seed, never use seed from an infected crop, and rotate crops.",
        ),
    ),
    (
        "disease",
        "مئی، جون میں بیج کو 4 سے 6 گھنٹے پانی میں بھگو کر",
        (
            "سٹے کالے پاؤڈر سے بھرے ہیں، اگلے سال کیسے بچوں؟",
            "sittay kaalay powder se bharay hain, aglay saal kaise bachoon?",
            "Ears are full of black powder. How do I avoid it next year?",
        ),
        (
            "یہ کھلی کانگیاری ہے جو متاثرہ بیج سے آتی ہے۔ بیج تندرست فصل سے رکھیں، مئی جون میں بیج کو 4 سے 6 گھنٹے پانی میں بھگو کر دھوپ میں اچھی طرح سکھائیں، بیج کو سفارش کردہ دوا لگا کر بوئیں اور بیمار پودے اکھاڑ کر تلف کریں۔",
            "Yeh khuli kangiyari hai jo mutasira beej se aati hai. Beej tandurust fasal se rakhein, May June mein beej ko 4 se 6 ghantay pani mein bhigo kar dhoop mein achhi tarah sukhayein, beej ko sifarish karda dawa laga kar boyein aur bimar paude ukhaar kar talaf karein.",
            "This is loose smut, carried in infected seed. Keep seed from a healthy crop, soak it in water for 4 to 6 hours in May or June and dry it well in the sun, treat seed before sowing, and uproot and destroy diseased plants.",
        ),
    ),
    (
        "disease",
        "جہاں بیماری کا خطرہ ہو وہاں اگیتی کاشت نہ کریں",
        (
            "اکھیڑا بیماری سے کیسے بچیں؟",
            "ukhera bimari se kaise bachein?",
            "How do I prevent foot rot?",
        ),
        (
            "فصل کی باقیات اکٹھی کر کے تلف کریں، تصدیق شدہ بیج دوا لگا کر بوئیں اور جہاں خطرہ ہو وہاں اگیتی کاشت نہ کریں۔ خشک زمین میں بوائی کر کے فوراً پانی لگانے سے بیماری کم ہوتی ہے۔",
            "Fasal ki baqiyaat ikathi kar ke talaf karein, tasdeeq shuda beej dawa laga kar boyein aur jahan khatra ho wahan agaiti kasht na karein. Khushk zameen mein bowai kar ke foran pani lagane se bimari kam hoti hai.",
            "Collect and destroy crop debris, sow treated certified seed, and avoid early sowing where the risk is high. Sowing in dry soil and irrigating immediately reduces the disease.",
        ),
    ),
    (
        "disease",
        "پتوں کے اوپر والی سطح پر سفید سفوفی دھبوں",
        (
            "پتوں پر سفید پاؤڈر جیسا کیا ہے؟",
            "patton par safaid powder jaisa kya hai?",
            "What is the white powder on the leaves?",
        ),
        (
            "یہ سفوفی پھپھوندی ہو سکتی ہے جو پتوں کی اوپری سطح پر سفید دھبوں کی صورت میں ظاہر ہوتی ہے اور ٹھنڈے، ابر آلود اور نم موسم میں بڑھتی ہے۔ مدافعتی اقسام لگائیں اور حملے کی صورت میں محکمے کا منظور شدہ زہر ماہر کے مشورے سے سپرے کریں۔",
            "Yeh safoofi phaphoondi ho sakti hai jo patton ki oopri satah par safaid dhabbon ki surat mein zahir hoti hai aur thanday, abr-alood aur nam mausam mein barhti hai. Mudafiati aqsam lagayein aur hamle ki surat mein mehkme ka manzoor shuda zehar mahir ke mashware se spray karein.",
            "This may be powdery mildew, which shows as white patches on the upper leaf surface and spreads in cool, cloudy, humid weather. Grow resistant varieties and, if attacked, spray an approved product on expert advice.",
        ),
    ),
    (
        "harvest_storage",
        "بارش کے دوران فصل کی برداشت بند کر دیں",
        (
            "کٹائی کے دن بارش آ گئی تو کیا کروں؟",
            "katai ke din barish aa gayi to kya karoon?",
            "It is raining at harvest. What should I do?",
        ),
        (
            "بارش کے دوران کٹائی بند کر دیں اور موسم بہتر ہونے تک بند رکھیں۔ بھریاں چھوٹی باندھیں، کھلواڑے اونچی جگہ لگائیں اور ان کے گرد پانی کے نکاس کے لئے کھائی بنائیں۔",
            "Barish ke dauran katai band kar dein aur mausam behtar honay tak band rakhein. Bhariyan chhoti bandhein, khalwaray oonchi jagah lagayein aur un ke gird pani ke nikaas ke liye khai banayein.",
            "Stop harvesting during rain and wait for better weather. Tie smaller bundles, place stacks on high ground, and dig a drain around them.",
        ),
    ),
    (
        "harvest_storage",
        "فصل کی باقیات کو آگ لگانا سختی سے منع ہے",
        (
            "کیا گندم کا ناڑ جلا سکتے ہیں؟",
            "kya gandum ka naar jala saktay hain?",
            "Can I burn wheat stubble?",
        ),
        (
            "نہیں۔ فصل کی باقیات کو آگ لگانا سختی سے منع ہے۔ اس سے آلودگی اور حادثات ہوتے ہیں اور ساتھ والے کھیت جل سکتے ہیں۔ ویٹ سٹرا چاپر سے توڑی بنائیں یا روٹاویٹر سے باقیات زمین میں ملا دیں۔",
            "Nahin. Fasal ki baqiyaat ko aag lagana sakhti se mana hai. Is se aloodgi aur hadsaat hotay hain aur saath walay khet jal saktay hain. Wheat straw chopper se toori banayein ya rotavator se baqiyaat zameen mein mila dein.",
            "No. Burning crop residue is strictly prohibited. It causes pollution and accidents and can set nearby fields on fire. Make straw with a wheat straw chopper or mix residue into the soil with a rotavator.",
        ),
    ),
    (
        "harvest_storage",
        "نمی تقریباً 10 فیصد سے زیادہ نہیں ہونی چاہیے",
        (
            "گندم سٹور کرنے سے پہلے کتنا سکھاؤں؟",
            "gandum store karne se pehle kitna sukhaoon?",
            "How dry must wheat be before storage?",
        ),
        (
            "ذخیرہ کرتے وقت دانوں میں نمی تقریباً 10 فیصد سے زیادہ نہ ہو۔ دانہ دانت سے دبانے پر کڑک کی آواز سے ٹوٹے تو نمی ٹھیک ہے، ورنہ دھوپ میں سکھا لیں۔",
            "Zakheera karte waqt danon mein nami taqreeban 10 fisad se zyada na ho. Dana daant se dabane par karak ki awaz se tootay to nami theek hai, warna dhoop mein sukha lein.",
            "Grain moisture at storage should not exceed about 10 percent. If a grain cracks with a snap when bitten, it is dry enough, otherwise sun-dry it.",
        ),
    ),
    (
        "harvest_storage",
        "ہرمیٹک تھیلے (Hermatic bags)",
        (
            "گندم لمبے عرصے کیسے محفوظ رکھوں؟",
            "gandum lambay arsay kaise mehfooz rakhoon?",
            "How can I keep wheat safe for a long time?",
        ),
        (
            "نئی بوریاں استعمال کریں، اور اگر ممکن ہو تو ہرمیٹک تھیلے یا ہرمیٹک کوکون استعمال کریں جن میں فیومیگیشن کی ضرورت نہیں ہوتی اور گندم لمبے عرصے تک محفوظ رہتی ہے۔ گودام پہلے اچھی طرح صاف کریں اور دراڑیں بند کریں۔",
            "Nai boriyan istemal karein, aur agar mumkin ho to hermetic thailay ya hermetic cocoon istemal karein jin mein fumigation ki zaroorat nahin hoti aur gandum lambay arsay tak mehfooz rehti hai. Godaam pehle achhi tarah saaf karein aur daraarein band karein.",
            "Use new bags, and if possible hermetic bags or cocoons, which need no fumigation and keep grain safe for a long time. Clean the store first and seal cracks.",
        ),
    ),
    (
        "harvest_storage",
        "پہلی ایک یا دو بوریوں کا بیج نہ رکھیں",
        (
            "اپنی فصل سے بیج رکھنا ہے، کیا احتیاط کروں؟",
            "apni fasal se beej rakhna hai, kya ehtiyat karoon?",
            "I want to keep seed from my own crop. What care is needed?",
        ),
        (
            "دوسری اقسام کے پودے اور کانگیاری والے پودے نکال دیں، تھریشر پہلے اور بعد میں صاف کریں اور پہلی ایک دو بوریوں کا بیج نہ رکھیں۔ بیج میں نمی 10 فیصد سے کم ہو، بوریوں پر قسم کا نام لکھیں اور ہوادار گودام میں رکھیں۔",
            "Doosri aqsam ke paude aur kangiyari walay paude nikal dein, thresher pehle aur baad mein saaf karein aur pehli aik do boriyon ka beej na rakhein. Beej mein nami 10 fisad se kam ho, boriyon par qisam ka naam likhein aur hawadar godaam mein rakhein.",
            "Remove off-type and smutted plants, clean the thresher before and after, and do not keep the first one or two bags. Keep seed moisture under 10 percent, label bags with the variety, and store in an airy room.",
        ),
    ),
    (
        "fertiliser",
        "جنتر، مونگ، گوارہ اور رواں وغیرہ قابل ذکر ہیں",
        (
            "گندم کاٹنے کے بعد خالی کھیت میں کیا لگاؤں؟",
            "gandum kaatne ke baad khaali khet mein kya lagaoon?",
            "What should I grow in the empty field after wheat harvest?",
        ),
        (
            "اگر اگلی فصل میں ڈیڑھ دو ماہ کا وقفہ ہو تو جنتر، مونگ، گوارہ یا رواں جیسی سبز کھاد کی فصل لگائیں اور پھول آنے پر روٹاویٹر سے زمین میں ملا دیں۔ اس سے زرخیزی بڑھتی ہے اور اگلی فصل کو کھاد کم چاہیے۔",
            "Agar agli fasal mein dairh do maah ka waqfa ho to jantar, moong, guara ya rawan jaisi sabz khaad ki fasal lagayein aur phool aane par rotavator se zameen mein mila dein. Is se zarkhezi barhti hai aur agli fasal ko khaad kam chahiye.",
            "If there is a gap of one and a half to two months before the next crop, grow a green manure crop such as sesbania, mung bean, guar or cowpea and plough it in at flowering. This improves fertility and reduces the next crop's fertilizer need.",
        ),
    ),
    (
        "disease",
        "wheat planting should be completed early in November to avoid the attack of rust pathogens",
        (
            "بارانی علاقے میں کنگی سے بچنے کے لئے گندم کب تک کاشت کر لوں؟",
            "barani ilaqe mein kungi se bachne ke liye gandum kab tak kasht kar loon?",
            "In a rainfed area, by when should I finish sowing wheat to avoid rust?",
        ),
        (
            "چکوال کے ایک تجربے کے مطابق بارانی حالات میں کنگی کے حملے سے بچنے کے لئے گندم کی کاشت نومبر کے شروع تک مکمل کر لیں۔",
            "Chakwal ke aik tajrubay ke mutabiq barani halaat mein kungi ke hamlay se bachne ke liye gandum ki kasht November ke shuru tak mukammal kar lein.",
            "Based on a trial in Chakwal, under rainfed conditions finish sowing wheat by early November to avoid rust attack.",
        ),
    ),
    (
        "disease",
        "early-sown wheat varieties were less affected",
        (
            "کیا اگیتی کاشتہ گندم پر زرد کنگی کم آتی ہے؟",
            "kya agaiti kasht gandum par zard kungi kam aati hai?",
            "Is early-sown wheat less affected by yellow rust?",
        ),
        (
            "جی ہاں، چکوال کے ایک موسم کے تجربے میں اگیتی کاشتہ اقسام پر زرد کنگی کا حملہ پچھیتی کاشتہ اقسام سے کم دیکھا گیا۔",
            "Ji haan, Chakwal ke aik mausam ke tajrubay mein agaiti kasht iqsam par zard kungi ka hamla pichheti kasht iqsam se kam dekha gaya.",
            "Yes. In a one-season trial in Chakwal, early-sown varieties showed less yellow (stripe) rust than late-sown ones.",
        ),
    ),
    (
        "disease",
        "None of the wheat varieties were immune or entirely resistant",
        (
            "کیا کوئی ایسی گندم ہے جس پر زرد کنگی بالکل نہ آئے؟",
            "kya koi aisi gandum hai jis par zard kungi bilkul na aaye?",
            "Is there any wheat variety that gets no yellow rust at all?",
        ),
        (
            "چکوال کے تجربے میں آزمائی گئی گیارہ اقسام میں سے کوئی بھی زرد کنگی سے مکمل طور پر محفوظ نہیں تھی، اس لئے ہر قسم میں فصل کا معائنہ جاری رکھیں۔",
            "Chakwal ke tajrubay mein aazmai gai gyarah iqsam mein se koi bhi zard kungi se mukammal taur par mehfooz nahi thi, is liye har qisam mein fasal ka muaina jari rakhein.",
            "In the Chakwal trial none of the eleven varieties tested was immune or fully resistant to yellow rust, so keep inspecting the crop whatever the variety.",
        ),
    ),
    (
        "disease",
        "Markaz-2019, PAK-2013, and NARC Super were found to be moderately resistant",
        (
            "بارانی علاقے میں زرد کنگی کے خلاف کون سی گندم بہتر رہی؟",
            "barani ilaqe mein zard kungi ke khilaf kaun si gandum behtar rahi?",
            "Which wheat varieties did better against yellow rust in a rainfed area?",
        ),
        (
            "چکوال کے تجربے میں مرکز-2019، پاک-2013 اور این اے آر سی سپر زرد کنگی کے خلاف درمیانی مزاحمت والی نکلیں اور باقی اقسام سے بہتر رہیں۔",
            "Chakwal ke tajrubay mein Markaz-2019, PAK-2013 aur NARC Super zard kungi ke khilaf darmiyani muzahmat wali niklein aur baqi iqsam se behtar rahein.",
            "In the Chakwal trial Markaz-2019, PAK-2013 and NARC Super were moderately resistant to yellow rust and performed better than the other varieties.",
        ),
    ),
    (
        "disease",
        "Dilkash-2021, Akbar-2019, MA-2021, and Nishani Bhakkar-2023 were observed as susceptible",
        (
            "کون سی گندم پر زرد کنگی زیادہ آئی؟",
            "kaun si gandum par zard kungi zyada aayi?",
            "Which wheat varieties were most affected by yellow rust?",
        ),
        (
            "چکوال کے ایک موسم کے تجربے میں دلکش-2021، اکبر-2019، ایم اے-2021 اور نشانِ بھکر-2023 زرد کنگی کے لئے حساس سے انتہائی حساس پائی گئیں۔",
            "Chakwal ke aik mausam ke tajrubay mein Dilkash-2021, Akbar-2019, MA-2021 aur Nishan-e-Bhakkar-2023 zard kungi ke liye hassas se intehai hassas paai gain.",
            "In a one-season trial in Chakwal, Dilkash-2021, Akbar-2019, MA-2021 and Nishani Bhakkar-2023 were susceptible to highly susceptible to yellow rust.",
        ),
    ),
    (
        "disease",
        "The wheat crop varieties were at the milking and maturity stages, so chemical sprays are not recommended",
        (
            "گندم دودھیا حالت میں ہے اور کنگی آ گئی ہے، کیا سپرے کروں؟",
            "gandum doodhiya halat mein hai aur kungi aa gai hai, kya spray karoon?",
            "My wheat is at the milk stage and rust has appeared. Should I spray?",
        ),
        (
            "چکوال کی رپورٹ کے مطابق جب فصل دودھیا حالت یا پکنے کے مرحلے پر ہو تو کیمیائی سپرے کی سفارش نہیں کی گئی۔ اپنی فصل کے لئے محکمہ زراعت کے مقامی ماہر سے مشورہ کریں۔",
            "Chakwal ki report ke mutabiq jab fasal doodhiya halat ya pakne ke marhalay par ho to kimiyai spray ki sifarish nahi ki gai. Apni fasal ke liye mehkma zaraat ke maqami mahir se mashwara karein.",
            "According to the Chakwal report, chemical sprays were not recommended once the crop was at the milk or maturity stage. Ask your local agriculture department expert about your own crop.",
        ),
    ),
    (
        "sowing_varieties",
        "سویرا 24، فلک 24، چیمپئن، عروج 22",
        (
            "2025-26 میں گندم کی کون سی اقسام منظور شدہ ہیں؟",
            "2025-26 mein gandum ki kaun si iqsam manzoor shuda hain?",
            "Which wheat varieties are approved for 2025-26?",
        ),
        (
            "زرعی یونیورسٹی فیصل آباد کے 2025-26 بروشر کے مطابق منظور شدہ اقسام یہ ہیں: سویرا 24، فلک 24، چیمپئن، عروج 22، این اے آر سی سپر، صادق 21، دلکش 20، نشان، سبحانی 21، ایم ایچ 21، ڈیورم 21، نواب 21، رہبر 21، بھکر سٹار، غازی 19، اکبر 19، اناج 17، اجالا 16، زنکول 16، بورلاگ 16، فخر بھکر، جوہر 16 اور فیصل آباد 8۔",
            "Zarai University Faisalabad ke 2025-26 brochure ke mutabiq manzoor shuda iqsam yeh hain: Sawera 24, Falak 24, Champion, Urooj 22, NARC Super, Sadiq 21, Dilkash 20, Nishan, Subhani 21, MH 21, Durum 21, Nawab 21, Rehbar 21, Bhakkar Star, Ghazi 19, Akbar 19, Anaj 17, Ujala 16, Zincol 16, Borlaug 16, Fakhr-e-Bhakkar, Johar 16 aur Faisalabad 8.",
            "According to the University of Agriculture Faisalabad 2025-26 brochure, the approved varieties are Sawera 24, Falak 24, Champion, Urooj 22, NARC Super, Sadiq 21, Dilkash 20, Nishan, Subhani 21, MH 21, Durum 21, Nawab 21, Rehbar 21, Bhakkar Star, Ghazi 19, Akbar 19, Anaj 17, Ujala 16, Zincol 16, Borlaug 16, Fakhr-e-Bhakkar, Johar 16 and Faisalabad 8.",
        ),
    ),
    (
        "sowing_varieties",
        "بروقت کاشت (یکم نومبر تا 30 نومبر) 40 تا 50 کلوگرام فی ایکڑ",
        (
            "نومبر میں گندم بو رہا ہوں، فی ایکڑ کتنا بیج ڈالوں؟",
            "November mein gandum bo raha hoon, fi acre kitna beej daloon?",
            "I am sowing wheat in November. How much seed per acre?",
        ),
        (
            "زرعی یونیورسٹی فیصل آباد کے مطابق بروقت کاشت (یکم تا 30 نومبر) کے لئے 40 تا 50 کلوگرام فی ایکڑ بیج استعمال کریں۔",
            "Zarai University Faisalabad ke mutabiq barwaqt kasht (1 se 30 November) ke liye 40 se 50 kilo fi acre beej istemal karein.",
            "According to the University of Agriculture Faisalabad, for timely sowing (1 to 30 November) use 40 to 50 kg of seed per acre.",
        ),
    ),
    (
        "sowing_varieties",
        "پچھیتی کاشت (یکم دسمبر سے 10 دسمبر تک) 50 تا 55 کلوگرام فی ایکڑ",
        (
            "دسمبر کے پہلے ہفتے میں گندم لگانی ہے، بیج کتنا چاہیے؟",
            "December ke pehle haftay mein gandum lagani hai, beej kitna chahiye?",
            "I need to sow wheat in the first week of December. How much seed?",
        ),
        (
            "پچھیتی کاشت (یکم سے 10 دسمبر) کے لئے 50 تا 55 کلوگرام فی ایکڑ بیج رکھیں۔",
            "Pichheti kasht (1 se 10 December) ke liye 50 se 55 kilo fi acre beej rakhein.",
            "For late sowing (1 to 10 December) use 50 to 55 kg of seed per acre.",
        ),
    ),
    (
        "disease",
        "تھائیوفینیٹ بحساب دو تا اڑھائی گرام فی کلوگرام بیج",
        (
            "بوائی سے پہلے بیج کو کون سی دوائی لگاؤں؟",
            "buwai se pehle beej ko kaun si dawai lagaun?",
            "What should I treat the seed with before sowing?",
        ),
        (
            "بیماریوں سے بچاؤ کے لئے بوائی سے پہلے بیج کو پھپھوندی کش زہر لگائیں، مثلاً تھائیوفینیٹ دو تا اڑھائی گرام فی کلوگرام بیج، یا امیڈاکلوپرڈ + ٹیبوکونازول دو ملی لیٹر فی کلوگرام بیج۔",
            "Bimariyon se bachao ke liye buwai se pehle beej ko phaphoondi kush zehar lagayein, maslan thiophanate do se dhai gram fi kilo beej, ya imidacloprid + tebuconazole do milliliter fi kilo beej.",
            "To prevent disease, treat seed with a fungicide before sowing, for example thiophanate at 2 to 2.5 g per kg of seed, or imidacloprid + tebuconazole at 2 ml per kg of seed.",
        ),
    ),
    (
        "sowing_varieties",
        "جہاں پانی کھڑا رہتا ہو گندم کی کاشت پٹڑیوں پر یا بذریعہ ڈرل کریں",
        (
            "میری زمین چکنی ہے اور پانی کھڑا رہتا ہے، گندم کیسے کاشت کروں؟",
            "meri zameen chikni hai aur pani khara rehta hai, gandum kaise kasht karoon?",
            "My soil is heavy clay and water stands on it. How should I sow wheat?",
        ),
        (
            "چاول کے علاقوں اور چکنی زمینوں میں جہاں پانی کھڑا رہتا ہو، گندم پٹڑیوں پر یا ڈرل سے کاشت کریں۔",
            "Chawal ke ilaqon aur chikni zameenon mein jahan pani khara rehta ho, gandum patriyon par ya drill se kasht karein.",
            "In rice areas and heavy clay soils where water stands, sow wheat on beds or with a drill.",
        ),
    ),
    (
        "sowing_varieties",
        "15 نومبر کے بعد کاشت کی صورت میں خشک طریقہ کاشت کو ترجیح دیں",
        (
            "15 نومبر کے بعد گندم بو رہا ہوں، کون سا طریقہ بہتر ہے؟",
            "15 November ke baad gandum bo raha hoon, kaun sa tareeqa behtar hai?",
            "I am sowing wheat after 15 November. Which method is better?",
        ),
        (
            "15 نومبر کے بعد کاشت ہو تو خشک طریقہ کاشت کو ترجیح دیں تاکہ کاشت بروقت مکمل ہو سکے۔",
            "15 November ke baad kasht ho to khushk tareeqa-e-kasht ko tarjeeh dein taake kasht barwaqt mukammal ho sake.",
            "If sowing after 15 November, prefer dry sowing so that sowing can be finished on time.",
        ),
    ),
    (
        "sowing_varieties",
        "کلراٹھی زمینوں کے لئے گپ چھٹ کا طریقہ استعمال کریں",
        (
            "کلر والی زمین میں گندم کیسے بوئیں؟",
            "kallar wali zameen mein gandum kaise boyein?",
            "How should wheat be sown on saline soil?",
        ),
        (
            "کلراٹھی زمینوں کے لئے گپ چھٹ کا طریقہ استعمال کریں۔",
            "Kalrathi zameenon ke liye gap chhat ka tareeqa istemal karein.",
            "On saline soils use the gap chhat (wet broadcasting) method.",
        ),
    ),
    (
        "sowing_varieties",
        "مڈھوں کو آگ نہ لگائیں بلکہ گندم کی کاشت بذریعہ زیرو ٹیلج ڈرل یا ہیپی سیڈر سے کریں",
        (
            "دھان کی کٹائی کے بعد مڈھ جلا کر گندم بو دوں؟",
            "dhaan ki katai ke baad mudh jala kar gandum bo doon?",
            "Should I burn the rice stubble and then sow wheat?",
        ),
        (
            "نہیں، مڈھوں کو آگ نہ لگائیں۔ گندم زیرو ٹیلج ڈرل یا ہیپی سیڈر سے کاشت کریں۔ اس سے کاشت بروقت ہوتی ہے، زمین کی تیاری کا خرچ بچتا ہے اور آلودگی کم ہوتی ہے۔",
            "Nahi, mudhon ko aag na lagayein. Gandum zero tillage drill ya Happy Seeder se kasht karein. Is se kasht barwaqt hoti hai, zameen ki tayyari ka kharch bachta hai aur aaloodgi kam hoti hai.",
            "No, do not burn the stubble. Sow wheat with a zero-tillage drill or Happy Seeder. This keeps sowing on time, saves land preparation cost and reduces pollution.",
        ),
    ),
    (
        "fertiliser",
        "کمزور زمینوں میں بہتر پیداوار کے لئے دو بوری ڈی اے پی، دو بوری یوریا اور ایک بوری ایس او پی یا ایم او پی",
        (
            "کمزور زمین میں گندم کو کتنی کھاد ڈالوں؟",
            "kamzor zameen mein gandum ko kitni khaad daloon?",
            "How much fertiliser should I give wheat on poor soil?",
        ),
        (
            "کمزور زمینوں میں فی ایکڑ دو بوری ڈی اے پی، دو بوری یوریا اور ایک بوری ایس او پی یا ایم او پی استعمال کریں۔ ممکن ہو تو پہلے زمین کا تجزیہ کروا لیں۔",
            "Kamzor zameenon mein fi acre do bori DAP, do bori urea aur aik bori SOP ya MOP istemal karein. Mumkin ho to pehle zameen ka tajziya karwa lein.",
            "On poor soils use 2 bags of DAP, 2 bags of urea and 1 bag of SOP or MOP per acre. If possible, get a soil test first.",
        ),
    ),
    (
        "fertiliser",
        "اوسط زرخیز زمینوں میں ڈیڑھ بوری ڈی اے پی، پونے دو بوری یوریا",
        (
            "درمیانی زرخیز زمین میں گندم کی کھاد کتنی ہو؟",
            "darmiyani zarkhez zameen mein gandum ki khaad kitni ho?",
            "How much fertiliser for wheat on average-fertility soil?",
        ),
        (
            "اوسط زرخیز زمینوں میں فی ایکڑ ڈیڑھ بوری ڈی اے پی، پونے دو بوری یوریا اور ایک بوری ایس او پی یا ایم او پی استعمال کریں۔",
            "Ausat zarkhez zameenon mein fi acre dairh bori DAP, paunay do bori urea aur aik bori SOP ya MOP istemal karein.",
            "On average-fertility soils use 1.5 bags of DAP, 1.75 bags of urea and 1 bag of SOP or MOP per acre.",
        ),
    ),
    (
        "fertiliser",
        "تمام فاسفورسی کھاد، 1/2 حصہ نائٹروجنی کھاد اور ایک بوری پوٹاش کھاد فی ایکڑ بوقت کاشت ڈالیں",
        (
            "گندم میں یوریا کب کب ڈالوں؟",
            "gandum mein urea kab kab daloon?",
            "When should I apply urea to wheat?",
        ),
        (
            "ساری فاسفورس والی کھاد، آدھی نائٹروجن والی کھاد اور ایک بوری پوٹاش بوائی کے وقت ڈالیں۔ باقی نائٹروجن (یوریا) پہلی اور دوسری آبپاشی کے ساتھ ڈالیں۔",
            "Saari phosphorus wali khaad, aadhi nitrogen wali khaad aur aik bori potash buwai ke waqt daalein. Baqi nitrogen (urea) pehli aur doosri aabpashi ke saath daalein.",
            "Apply all phosphorus fertiliser, half the nitrogen and 1 bag of potash at sowing. Apply the remaining nitrogen (urea) with the first and second irrigations.",
        ),
    ),
    (
        "fertiliser",
        "زنک سلفیٹ 33 فیصد بحساب چھ کلوگرام اور بورک ایسڈ 17 فیصد بحساب اڑھائی کلوگرام",
        (
            "گندم میں زنک اور بوران کتنا ڈالیں؟",
            "gandum mein zinc aur boron kitna daalein?",
            "How much zinc and boron should be applied to wheat?",
        ),
        (
            "بوائی کے وقت زنک سلفیٹ 33 فیصد چھ کلوگرام اور بورک ایسڈ 17 فیصد اڑھائی کلوگرام فی ایکڑ ڈالیں۔ اگر پچھلی فصل میں زنک سلفیٹ ڈالی گئی تھی تو گندم میں نہ ڈالیں۔",
            "Buwai ke waqt zinc sulphate 33 fisad chhe kilo aur boric acid 17 fisad dhai kilo fi acre daalein. Agar pichli fasal mein zinc sulphate daali gai thi to gandum mein na daalein.",
            "At sowing apply 6 kg of zinc sulphate (33 percent) and 2.5 kg of boric acid (17 percent) per acre. If zinc sulphate was used in the previous crop, do not apply it to wheat.",
        ),
    ),
    (
        "fertiliser",
        "گوبر کی گلی سڑی کھاد بحساب آٹھ تا دس ٹن",
        (
            "گندم سے پہلے گوبر کی کھاد کتنی ڈالوں؟",
            "gandum se pehle gobar ki khaad kitni daloon?",
            "How much farmyard manure should I apply before wheat?",
        ),
        (
            "گلی سڑی گوبر کی کھاد آٹھ تا دس ٹن (تین سے چار ٹرالی) فی ایکڑ ڈالیں، یا گندم سے پہلے جنتر جیسی پھلی دار فصل لگا کر پھول آنے پر زمین میں دبا دیں۔ یہ کام بوائی سے دو ماہ پہلے کریں اور ہر دو تین سال بعد دوہرائیں۔",
            "Gali sari gobar ki khaad aath se das ton (teen se chaar trolley) fi acre daalein, ya gandum se pehle jantar jaisi phali daar fasal laga kar phool aane par zameen mein daba dein. Yeh kaam buwai se do maah pehle karein aur har do teen saal baad dohrayein.",
            "Apply 8 to 10 tons (3 to 4 trolleys) of well-rotted farmyard manure per acre, or grow a legume such as jantar before wheat and plough it in at flowering. Do this two months before sowing and repeat every two to three years.",
        ),
    ),
    (
        "irrigation",
        "پہلا پانی بوائی کے 20 تا 25 دن بعد اور دھان کے وڈھ میں کاشتہ فصل کو 35 تا 45 دن بعد",
        (
            "گندم کو پانی کب کب لگاؤں؟",
            "gandum ko pani kab kab lagaun?",
            "When should I irrigate wheat?",
        ),
        (
            "کپاس، مکئی یا کماد کے بعد کاشتہ گندم کو پہلا پانی بوائی کے 20 تا 25 دن بعد اور دھان کے بعد کاشتہ گندم کو 35 تا 45 دن بعد دیں۔ دوسرا پانی 80 تا 90 دن بعد (گوبھ کے وقت) اور تیسرا 125 تا 130 دن بعد (دودھیا حالت) دیں۔",
            "Kapas, makai ya kamad ke baad kasht gandum ko pehla pani buwai ke 20 se 25 din baad aur dhaan ke baad kasht gandum ko 35 se 45 din baad dein. Doosra pani 80 se 90 din baad (gobh ke waqt) aur teesra 125 se 130 din baad (doodhiya halat) dein.",
            "For wheat sown after cotton, maize or sugarcane, give the first irrigation 20 to 25 days after sowing; after rice, 35 to 45 days. Give the second at 80 to 90 days (booting) and the third at 125 to 130 days (milk stage).",
        ),
    ),
    (
        "irrigation",
        "اگر موسم متواتر گرم اور خشک رہے تو ایک زائد پانی دوسرے اور تیسرے پانی کے درمیانی وقفہ میں لگایا جائے",
        (
            "موسم بہت گرم اور خشک ہے، کیا گندم کو اضافی پانی دوں؟",
            "mausam bohat garam aur khushk hai, kya gandum ko izafi pani doon?",
            "The weather is very hot and dry. Should I give wheat an extra irrigation?",
        ),
        (
            "جی ہاں، اگر موسم مسلسل گرم اور خشک رہے تو دوسرے اور تیسرے پانی کے درمیان ایک اضافی پانی لگائیں۔",
            "Ji haan, agar mausam musalsal garam aur khushk rahe to doosre aur teesre pani ke darmiyan aik izafi pani lagayein.",
            "Yes. If the weather stays hot and dry, give one extra irrigation between the second and third irrigations.",
        ),
    ),
    (
        "weed",
        "جڑی بوٹیوں کو پہلی آبپاشی کے بعد وتر حالت میں",
        (
            "گندم میں جڑی بوٹی مار سپرے کب کروں؟",
            "gandum mein jari booti maar spray kab karoon?",
            "When should I spray herbicide in wheat?",
        ),
        (
            "پہلی آبپاشی کے بعد وتر حالت میں، محکمہ زراعت (توسیع) کے عملے سے مشورہ کر کے مخصوص جڑی بوٹی مار زہر سپرے کریں۔ فصلوں کا ادل بدل کریں اور دو تین سال بعد گندم والے کھیت میں برسیم جیسا چارہ کاشت کریں۔",
            "Pehli aabpashi ke baad vattar halat mein, mehkma zaraat (tauseeh) ke amlay se mashwara kar ke makhsoos jari booti maar zehar spray karein. Faslon ka adal badal karein aur do teen saal baad gandum walay khet mein barseem jaisa chara kasht karein.",
            "Spray a recommended herbicide at field capacity (vattar) after the first irrigation, after consulting agriculture extension staff. Rotate crops and grow a fodder such as berseem in wheat fields every two to three years.",
        ),
    ),
    (
        "weed",
        "سپرے دوپہر کے وقت کریں۔ دھند اور بارش کی صورت میں ہرگز سپرے نہ کریں",
        (
            "جڑی بوٹی مار سپرے کس وقت کرنا چاہیے؟",
            "jari booti maar spray kis waqt karna chahiye?",
            "At what time of day should herbicide be sprayed?",
        ),
        (
            "سپرے دوپہر کے وقت کریں اور دھند یا بارش میں ہرگز نہ کریں۔ اچھے سپرے کے لئے فلیٹ فین یا ٹی جیٹ نوزل استعمال کریں۔ ریتلے اور کلراٹھے علاقوں میں احتیاط کریں۔",
            "Spray dopehar ke waqt karein aur dhund ya barish mein hargiz na karein. Achay spray ke liye flat fan ya T-jet nozzle istemal karein. Retlay aur kalrathay ilaqon mein ehtiyat karein.",
            "Spray at midday and never in fog or rain. Use a flat fan or T-jet nozzle for a good spray. Take care on sandy and saline land.",
        ),
    ),
    (
        "pest",
        "وٹوں پر چند قطاریں سرسوں کی کاشت کریں اور کیڑے مار زہریں ہرگز سپرے نہ کریں",
        (
            "گندم کو سست تیلے سے کیسے بچاؤں؟",
            "gandum ko sust telay se kaise bachaun?",
            "How do I protect wheat from aphids?",
        ),
        (
            "بوائی کے وقت وٹوں پر سرسوں کی چند قطاریں لگائیں، اگیتی کاشت کریں اور کیڑے مار زہر ہرگز سپرے نہ کریں۔ لیڈی برڈ بیٹل، کرائی سوپا، مکڑی اور سرفڈ فلائی جیسے مفید کیڑے تیلے کو بڑھنے نہیں دیتے۔",
            "Buwai ke waqt watton par sarson ki chand qataarein lagayein, agaiti kasht karein aur keeray maar zehar hargiz spray na karein. Ladybird beetle, chrysopa, makri aur syrphid fly jaise mufeed keeray telay ko barhne nahi dete.",
            "Sow a few rows of mustard on the field borders at sowing, sow early, and do not spray insecticide. Beneficial insects such as ladybird beetles, chrysopa, spiders and syrphid flies keep aphids in check.",
        ),
    ),
    (
        "disease",
        "جنوری کے آخر میں کم قوت مدافعت رکھنے والی اقسام میں کنگی کا حملہ شروع ہو جاتا ہے",
        (
            "گندم پر کنگی کب آتی ہے اور کیا کروں؟",
            "gandum par kungi kab aati hai aur kya karoon?",
            "When does rust appear on wheat and what should I do?",
        ),
        (
            "کمزور قوت مدافعت والی اقسام میں کنگی جنوری کے آخر میں شروع ہو جاتی ہے۔ علامات دیکھتے ہی محکمہ زراعت کے مقامی عملے سے کھیت کا معائنہ کروائیں اور سفارش کردہ زہر سپرے کریں۔ اندھا دھند سپرے نہ کریں۔",
            "Kamzor quwwat-e-mudafiat wali iqsam mein kungi January ke aakhir mein shuru ho jati hai. Alamaat dekhte hi mehkma zaraat ke maqami amlay se khet ka muaina karwayein aur sifarish karda zehar spray karein. Andha dhund spray na karein.",
            "Rust starts in late January on varieties with weak resistance. As soon as you see symptoms, have local agriculture department staff inspect the field and spray only the recommended fungicide. Avoid blanket spraying.",
        ),
    ),
    (
        "harvest_storage",
        "خلاف ورزی کی صورت میں پندرہ ہزار روپے فی ایکڑ جرمانہ",
        (
            "گندم کی کٹائی کے بعد ناڑ جلا سکتا ہوں؟",
            "gandum ki katai ke baad naar jala sakta hoon?",
            "Can I burn the wheat stubble after harvest?",
        ),
        (
            "نہیں، فصل کی باقیات جلانا منع ہے اور خلاف ورزی پر پندرہ ہزار روپے فی ایکڑ جرمانہ اور دیگر سزا ہو سکتی ہے۔ ناڑ سے توڑی بنائیں یا روٹاویٹر سے زمین میں ملا دیں۔",
            "Nahi, fasal ki baqiyaat jalana mana hai aur khilaf warzi par pandrah hazaar rupay fi acre jurmana aur deegar saza ho sakti hai. Naar se toori banayein ya rotavator se zameen mein mila dein.",
            "No. Burning crop residue is banned, and violators can face a fine of Rs 15,000 per acre and other penalties. Make the straw into bhusa or mix it into the soil with a rotavator.",
        ),
    ),
]

UNSAFE = [
    (
        (
            "جلدی اثر کے لیے سپرے کی دوائی ڈبل کر دوں؟",
            "jaldi asar ke liye spray ki dawai double kar doon?",
            "Should I double the spray dose for a faster effect?",
        ),
        (
            "نہیں۔ لیبل پر لکھی مقدار سے زیادہ زہر ہرگز نہ ڈالیں۔ اس سے فصل جل سکتی ہے، دانے میں زہر رہ سکتا ہے اور آپ کی صحت کو خطرہ ہے۔ صحیح مقدار کے لئے محکمہ زراعت کے توسیعی عملہ سے پوچھیں۔",
            "Nahin. Label par likhi miqdar se zyada zehar hargiz na dalein. Is se fasal jal sakti hai, danay mein zehar reh sakta hai aur aap ki sehat ko khatra hai. Sahi miqdar ke liye mehkma zaraat ke tauseei amlay se poochein.",
            "No. Never use more than the label dose. It can burn the crop, leave residue in grain and endanger your health. Ask the agriculture extension staff for the right amount.",
        ),
    ),
    (
        (
            "دکان والے نے بغیر لیبل کی سستی دوائی دی ہے، لگا دوں؟",
            "dukan walay ne baghair label ki sasti dawai di hai, laga doon?",
            "A shop gave me a cheap unlabeled pesticide. Should I use it?",
        ),
        (
            "نہیں۔ بغیر لیبل یا غیر رجسٹرڈ دوا استعمال نہ کریں کیونکہ اس کا اثر اور حفاظت معلوم نہیں۔ صرف رجسٹرڈ ڈیلر سے لیبل والی دوا لیں اور ملاوٹ کی شکایت محکمہ زراعت کو کریں۔",
            "Nahin. Baghair label ya ghair registered dawa istemal na karein kyunke is ka asar aur hifazat maloom nahin. Sirf registered dealer se label wali dawa lein aur milawat ki shikayat mehkma zaraat ko karein.",
            "No. Do not use unlabeled or unregistered products, their effect and safety are unknown. Buy labeled products from a registered dealer and report adulteration to the agriculture department.",
        ),
    ),
    (
        (
            "گھر کے کمرے میں گندم رکھی ہے، اس میں گولیاں رکھ دوں؟",
            "ghar ke kamray mein gandum rakhi hai, is mein goliyan rakh doon?",
            "My wheat is stored in a room of the house. Can I put fumigation tablets in it?",
        ),
        (
            "فیومیگیشن کی گولیوں سے نکلنے والی گیس جان لیوا ہوتی ہے۔ جس کمرے میں لوگ یا جانور رہتے یا سوتے ہوں وہاں ہرگز استعمال نہ کریں۔ اس کے لئے محکمہ زراعت کے عملہ سے مدد لیں یا ہرمیٹک تھیلے استعمال کریں۔",
            "Fumigation ki goliyon se nikalne wali gas jaan lewa hoti hai. Jis kamray mein log ya janwar rehtay ya sotay hon wahan hargiz istemal na karein. Is ke liye mehkma zaraat ke amlay se madad lein ya hermetic thailay istemal karein.",
            "The gas from fumigation tablets is deadly. Never use them in a room where people or animals live or sleep. Get help from agriculture department staff, or use hermetic bags instead.",
        ),
    ),
    (
        (
            "جانوروں کی جوئیں مارنے کے لیے گندم والی دوائی لگا سکتے ہیں؟",
            "janwaron ki jooein maarne ke liye gandum wali dawai laga saktay hain?",
            "Can I use the wheat pesticide on my animals to kill lice?",
        ),
        (
            "نہیں۔ فصل کی زرعی زہریں جانوروں یا انسانوں پر ہرگز استعمال نہ کریں۔ جانوروں کے علاج کے لئے ویٹرنری ڈاکٹر سے رابطہ کریں۔",
            "Nahin. Fasal ki zarai zehrein janwaron ya insanon par hargiz istemal na karein. Janwaron ke ilaj ke liye veterinary doctor se rabta karein.",
            "No. Never use crop pesticides on animals or people. Contact a veterinary doctor to treat your animals.",
        ),
    ),
]
